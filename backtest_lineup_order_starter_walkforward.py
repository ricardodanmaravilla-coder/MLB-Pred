"""Strict validation: confirmed batting order + real starter-hand matchup + prior starter quality.
Uses only information dated strictly before each game. Production is untouched.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SRC=Path('artifacts/lineup_platoon_walkforward_predictions.csv'); OUT=Path('artifacts/lineup_order_starter_predictions.csv'); REPORT=Path('artifacts/lineup_order_starter_report.json')
FEATURES=['order_ops_hand_diff','order_bb_hand_diff','order_k_hand_diff','starter_era_diff','starter_whip_diff','starter_kbb_diff']
C_GRID=(.003,.005,.01,.03); MIN_PA=25.; MIN_START_IP=20.; ORDER_W=np.array([1.12,1.08,1.05,1.03,1.00,.97,.94,.92,.89])
STAT=['AB','H','2B','3B','HR','BB','SO','HBP','SF']

def logit(p): p=np.clip(np.asarray(p,float),1e-5,1-1e-5); return np.log(p/(1-p))
def design(d): return np.c_[logit(d.baseline_home_prob),d[FEATURES].to_numpy(float)]
def brier(y,p): return float(np.mean((np.asarray(y)-np.asarray(p))**2))
def ll(y,p): p=np.clip(np.asarray(p),1e-9,1-1e-9); y=np.asarray(y); return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
def fit(d,c):
    m=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=2000,random_state=42)); m.fit(design(d),d.actual_home_win.astype(int)); return m

def hitter_profile(h,pid,hand,date):
    z=h[(h.PlayerID==pid)&(h.PitcherHand==hand)&(h.Date<date)&(h.Date>=date-pd.Timedelta(days=365))]
    if z.empty:return None
    s=z[STAT].sum(); pa=s.AB+s.BB+s.HBP+s.SF
    if pa<MIN_PA or s.AB<=0:return None
    tb=s.H+s['2B']+2*s['3B']+3*s.HR; obp=(s.H+s.BB+s.HBP)/pa; slg=tb/s.AB
    return obp+slg,s.BB/pa,s.SO/pa

def starter_profile(q,pid,date):
    z=q[(q.StarterID==pid)&(q.Date<date)&(q.Date>=date-pd.Timedelta(days=365))]
    if z.empty:return None
    s=z[['IP','H','ER','BB','SO','HR','BF']].sum(); ip=s.IP
    if ip<MIN_START_IP:return None
    era=9*s.ER/ip; whip=(s.H+s.BB)/ip; kbb=(s.SO-s.BB)/max(s.BF,1)
    return era,whip,kbb

def enrich(src):
    lu=pd.read_csv('data/mlb_lineup_history.csv',low_memory=False); hp=pd.read_csv('data/mlb_hitter_platoon_game_history.csv',low_memory=False); st=pd.read_csv('data/mlb_starter_hand_history.csv',low_memory=False); q=pd.read_csv('data/mlb_starter_quality_history.csv',low_memory=False)
    for x in (hp,q):x['Date']=pd.to_datetime(x.Date,errors='coerce')
    hp.PlayerID=pd.to_numeric(hp.PlayerID,errors='coerce'); hp.PitcherHand=hp.PitcherHand.astype(str).str.upper(); [hp.__setitem__(c,pd.to_numeric(hp[c],errors='coerce').fillna(0)) for c in STAT]
    q.StarterID=pd.to_numeric(q.StarterID,errors='coerce'); [q.__setitem__(c,pd.to_numeric(q[c],errors='coerce').fillna(0)) for c in ['IP','H','ER','BB','SO','HR','BF']]
    lu.GameID=pd.to_numeric(lu.GameID,errors='coerce'); lu.PlayerID=pd.to_numeric(lu.PlayerID,errors='coerce'); st.GameID=pd.to_numeric(st.GameID,errors='coerce'); st.StarterID=pd.to_numeric(st.StarterID,errors='coerce'); st.PitchHand=st.PitchHand.astype(str).str.upper()
    lg={(int(g),s):x.sort_values('BattingOrder').head(9) for (g,s),x in lu.dropna(subset=['GameID','PlayerID']).groupby(['GameID','Side'])}; sg={(int(r.GameID),str(r.Side)):(int(r.StarterID),r.PitchHand) for _,r in st.dropna(subset=['GameID','StarterID']).iterrows()}
    rows=[]
    for _,r in src.iterrows():
        gid=int(r.GameID); d=pd.Timestamp(r.Date); vals={}; valid=True; sidep={}
        for side,opp in [('home','away'),('away','home')]:
            oppst=sg.get((gid,opp)); ownst=sg.get((gid,side)); lineup=lg.get((gid,side)); prof=[]
            if not oppst or lineup is None: valid=False; break
            for j,pid in enumerate(lineup.PlayerID.astype(int)):
                p=hitter_profile(hp,pid,oppst[1],d)
                if p is not None:prof.append((j,p))
            if len(prof)<7:valid=False; break
            ww=np.array([ORDER_W[j] for j,_ in prof]); ww/=ww.sum(); sidep[side]=[float(sum(w*p[k] for w,(_,p) in zip(ww,prof))) for k in range(3)]
            sp=starter_profile(q,ownst[0],d) if ownst else None
            if sp is None:valid=False; break
            sidep[side+'st']=sp
        if valid:
            vals={'order_ops_hand_diff':sidep['home'][0]-sidep['away'][0],'order_bb_hand_diff':sidep['home'][1]-sidep['away'][1],'order_k_hand_diff':sidep['home'][2]-sidep['away'][2],'starter_era_diff':sidep['awayst'][0]-sidep['homest'][0],'starter_whip_diff':sidep['awayst'][1]-sidep['homest'][1],'starter_kbb_diff':sidep['homest'][2]-sidep['awayst'][2]}
        rows.append(vals)
    return pd.concat([src.reset_index(drop=True),pd.DataFrame(rows)],axis=1)

def main():
    s=pd.read_csv(SRC); s.Date=pd.to_datetime(s.Date,errors='coerce'); x=enrich(s); x=x.dropna(subset=FEATURES).sort_values('Date').reset_index(drop=True); coverage=len(x)/max(len(s),1)
    if len(x)<1200:raise SystemExit(f'Insufficient combined sample {len(x)} coverage={coverage:.1%}')
    # Preserve the exact outer folds from the previous strict test; select regularization only on prior data.
    pooled=[]; reps=[]
    for fold in sorted(x['fold'].dropna().unique()):
        te=x[x['fold']==fold].copy(); start=te.Date.min(); tr=x[x.Date<start].copy()
        if len(tr)<500 or len(te)<200:continue
        days=pd.Series(tr.Date.dt.normalize().unique()).sort_values().reset_index(drop=True); cut=pd.Timestamp(days.iloc[int(len(days)*.75)]); a=tr[tr.Date.dt.normalize()<cut]; v=tr[tr.Date.dt.normalize()>=cut]; trials=[]
        for c in C_GRID:
            m=fit(a,c); p=m.predict_proba(design(v))[:,1]; trials.append((brier(v.actual_home_win,p)+.15*ll(v.actual_home_win,p),c))
        c=min(trials)[1]; m=fit(tr,c); te['candidate_home_prob']=m.predict_proba(design(te))[:,1]; te['candidate_total_runs']=te.baseline_total_runs; pooled.append(te); reps.append({'fold':int(fold),'train_rows':len(tr),'test_rows':len(te),'C':c})
    if not pooled:raise SystemExit('No valid combined folds')
    out=pd.concat(pooled).sort_values('Date'); OUT.parent.mkdir(exist_ok=True); out.to_csv(OUT,index=False); rep={'policy':'confirmed_order_platoon_plus_prior_starter_quality_v1','rows':len(out),'source_rows':len(s),'coverage':coverage,'features':FEATURES,'folds':reps}; REPORT.write_text(json.dumps(rep,indent=2)); print('ORDER_STARTER_WALKFORWARD',json.dumps(rep)); print(f'OK {OUT} rows={len(out)} coverage={coverage:.1%}')
if __name__=='__main__':main()
