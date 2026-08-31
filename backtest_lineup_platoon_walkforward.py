"""Strict nested walk-forward test: confirmed lineup x real opposing starter hand.

Every split profile is built from official MLB plate appearances strictly before the
game date. Same-day and future results are excluded. Missing starter hand or insufficient
hitter-vs-hand history makes that game ineligible instead of creating a proxy.
Validation-only: production code is untouched.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from modules.historical_mlb import prepare_games
from modules.metric_quality import batting_metric,pitching_metric
from modules.ml_mlb import PredictorMLMLB
from modules.team_utils import normalize_team

OUT=Path('artifacts/lineup_platoon_walkforward_predictions.csv')
REPORT=Path('artifacts/lineup_platoon_walkforward_report.json')
STAT_COLS=('AB','H','2B','3B','HR','BB','SO','HBP','SF')
FEATURES=('lu_ops_vs_hand_diff','lu_bb_vs_hand_diff','lu_k_vs_hand_diff')
C_GRID=(0.005,0.01,0.03)
MIN_HAND_PA=25.0
MIN_QUALIFIED=7


def _date_boundary(games,frac=.70):
    d=pd.Series(games.Date.dt.normalize().dropna().unique()).sort_values().reset_index(drop=True)
    return pd.Timestamp(d.iloc[max(1,min(len(d)-1,int(len(d)*frac)))])

def _maps(df,m):
    x=df.copy(); x['Team']=x.Team.map(normalize_team); x['Season']=pd.to_numeric(x.Season,errors='coerce'); x[m]=pd.to_numeric(x[m],errors='coerce')
    return x.dropna(subset=['Team','Season',m]).set_index(['Team','Season'])[m].to_dict()

def _logit(p):
    p=np.clip(np.asarray(p,float),1e-5,1-1e-5); return np.log(p/(1-p))

def _design(df): return np.hstack([_logit(df.baseline_home_prob).reshape(-1,1),df[list(FEATURES)].to_numpy(float)])
def _brier(y,p): return float(np.mean((np.asarray(p)-np.asarray(y))**2))
def _ll(y,p):
    p=np.clip(np.asarray(p),1e-9,1-1e-9); y=np.asarray(y); return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
def _ece(y,p):
    y=np.asarray(y); p=np.asarray(p); z=0.; e=np.linspace(0,1,11)
    for i in range(10):
        m=(p>=e[i])&((p<e[i+1]) if i<9 else (p<=e[i+1])); z+=(m.sum()/len(y))*abs(float(p[m].mean()-y[m].mean())) if m.any() else 0
    return float(z)
def _fit(df,c):
    m=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=2000,solver='lbfgs',random_state=42)); m.fit(_design(df),df.actual_home_win.astype(int)); return m


class PlatoonRolling365:
    def __init__(self,path):
        h=pd.read_csv(path,low_memory=False); h['Date']=pd.to_datetime(h.Date,errors='coerce'); h['PlayerID']=pd.to_numeric(h.PlayerID,errors='coerce'); h['PitcherHand']=h.PitcherHand.astype(str).str.upper()
        for c in STAT_COLS: h[c]=pd.to_numeric(h.get(c),errors='coerce').fillna(0.)
        h=h.dropna(subset=['Date','PlayerID']); h=h[h.PitcherHand.isin(['R','L'])].sort_values(['PlayerID','PitcherHand','Date','GameID'])
        self.data={}
        for (pid,hand),g in h.groupby([h.PlayerID.astype(int),'PitcherHand'],sort=False):
            dates=g.Date.dt.normalize().to_numpy(dtype='datetime64[ns]'); vals=g[list(STAT_COLS)].to_numpy(float); cs=np.vstack([np.zeros((1,len(STAT_COLS))),np.cumsum(vals,axis=0)])
            self.data[(int(pid),hand)]=(dates,cs)
    def profile(self,pid,hand,date):
        z=self.data.get((int(pid),str(hand).upper()))
        if z is None:return None
        dates,cs=z; d=np.datetime64(pd.Timestamp(date).normalize()); lo=np.searchsorted(dates,d-np.timedelta64(365,'D'),'left'); hi=np.searchsorted(dates,d,'left')
        if hi<=lo:return None
        s=dict(zip(STAT_COLS,cs[hi]-cs[lo])); pa=s['AB']+s['BB']+s['HBP']+s['SF']
        if pa<MIN_HAND_PA or s['AB']<=0:return None
        tb=s['H']+s['2B']+2*s['3B']+3*s['HR']; obp=(s['H']+s['BB']+s['HBP'])/max(pa,1.); slg=tb/max(s['AB'],1.)
        return {'ops':obp+slg,'bb':s['BB']/pa,'k':s['SO']/pa,'pa':pa}


def _features(games,lineups,starters,roller):
    lu=lineups.copy(); lu['GameID']=pd.to_numeric(lu.GameID,errors='coerce'); lu['PlayerID']=pd.to_numeric(lu.PlayerID,errors='coerce')
    lg={(int(g),s):x.sort_values('BattingOrder') for (g,s),x in lu.dropna(subset=['GameID','PlayerID']).groupby(['GameID','Side'])}
    st=starters.copy(); st['GameID']=pd.to_numeric(st.GameID,errors='coerce'); st['PitchHand']=st.PitchHand.astype(str).str.upper()
    sm={(int(r.GameID),str(r.Side)):r.PitchHand for _,r in st.dropna(subset=['GameID']).iterrows() if r.PitchHand in ('R','L')}
    rows=[]
    for _,r in games.iterrows():
        gid=int(r.GameID) if pd.notna(r.get('GameID')) else int(r.gamePk); hh=sm.get((gid,'home')); ah=sm.get((gid,'away')); sides={}
        for side,opp_hand in (('home',ah),('away',hh)):
            prof=[]; x=lg.get((gid,side))
            if x is not None and opp_hand in ('R','L'):
                for pid in x.PlayerID.astype(int).head(9):
                    p=roller.profile(pid,opp_hand,r.Date)
                    if p is not None: prof.append(p)
            sides[side]={'qualified':len(prof),'ops':float(np.mean([p['ops'] for p in prof])) if prof else np.nan,'bb':float(np.mean([p['bb'] for p in prof])) if prof else np.nan,'k':float(np.mean([p['k'] for p in prof])) if prof else np.nan}
        valid=hh in ('R','L') and ah in ('R','L') and sides['home']['qualified']>=MIN_QUALIFIED and sides['away']['qualified']>=MIN_QUALIFIED
        rows.append({'home_starter_hand':hh,'away_starter_hand':ah,'home_platoon_qualified':sides['home']['qualified'],'away_platoon_qualified':sides['away']['qualified'],'lu_ops_vs_hand_diff':sides['home']['ops']-sides['away']['ops'] if valid else np.nan,'lu_bb_vs_hand_diff':sides['home']['bb']-sides['away']['bb'] if valid else np.nan,'lu_k_vs_hand_diff':sides['home']['k']-sides['away']['k'] if valid else np.nan,'platoon_valid':float(valid)})
    return pd.DataFrame(rows,index=games.index)


def _select(train):
    days=pd.Series(train.Date.dt.normalize().unique()).sort_values().reset_index(drop=True); cut=pd.Timestamp(days.iloc[max(1,min(len(days)-1,int(len(days)*.75)))]); tr=train[train.Date.dt.normalize()<cut]; va=train[train.Date.dt.normalize()>=cut]
    y=va.actual_home_win.astype(int); be=_ece(y,va.baseline_home_prob); trials=[]
    for c in C_GRID:
        m=_fit(tr,c); p=m.predict_proba(_design(va))[:,1]; b,l,e=_brier(y,p),_ll(y,p),_ece(y,p); score=b+.15*l+.10*max(0,e-be)
        trials.append({'C':c,'features':list(FEATURES),'score':score,'brier':b,'log_loss':l,'ece':e})
    trials.sort(key=lambda x:(x['score'],x['brier'],x['log_loss'])); return trials[0],trials,str(cut.date())


def main():
    bat=pd.read_csv('data/mlb_batting.csv'); pit=pd.read_csv('data/mlb_pitching.csv'); games=prepare_games(pd.read_csv('data/mlb_games.csv')); boundary=_date_boundary(games); train=games[games.Date.dt.normalize()<boundary]; later=games[games.Date.dt.normalize()>=boundary]
    model=PredictorMLMLB(); assert model.entrenar(bat,pit,train); bc,pc=batting_metric(bat),pitching_metric(pit); bd,pdct=_maps(bat,bc),_maps(pit,pc); bmed=float(pd.to_numeric(bat[bc],errors='coerce').median()); pmed=float(pd.to_numeric(pit[pc],errors='coerce').median()); rows=[]
    for day,dg in later.groupby(later.Date.dt.normalize(),sort=True):
        pending=[]
        for idx,r in dg.iterrows():
            h,a=normalize_team(r.Home),normalize_team(r.Away); sy=int(r.Season)-1; pred=model.predecir_partido(h,a,float(bd.get((h,sy),bmed)),float(bd.get((a,sy),bmed)),float(pdct.get((h,sy),pmed)),float(pdct.get((a,sy),pmed)),game_date=r.Date); hs,aw=float(r.Home_Score),float(r.Away_Score)
            rows.append({'idx':idx,'Date':r.Date,'GameID':r.get('GameID',r.get('gamePk')),'actual_home_win':int(hs>aw),'actual_total_runs':hs+aw,'baseline_home_prob':float(pred['Probabilidad_Local'])/100.,'baseline_total_runs':float(pred['Proyeccion_Carreras'])}); pending.append((h,a,hs,aw,r.Date))
        for x in pending:model.actualizar_resultado(x[0],x[1],x[2],x[3],game_date=x[4])
    frame=pd.DataFrame(rows).set_index('idx'); lineups=pd.read_csv('data/mlb_lineup_history.csv',low_memory=False); starters=pd.read_csv('data/mlb_starter_hand_history.csv',low_memory=False); roller=PlatoonRolling365('data/mlb_hitter_platoon_game_history.csv')
    frame=frame.join(_features(later.loc[frame.index],lineups,starters,roller)).sort_values('Date').reset_index(drop=True); frame.Date=pd.to_datetime(frame.Date); source=len(frame); valid=frame[(frame.platoon_valid>=1)&frame[list(FEATURES)].notna().all(axis=1)].copy(); coverage=len(valid)/max(source,1)
    if len(valid)<900: raise SystemExit(f'Insufficient qualified platoon sample: {len(valid)} ({coverage:.1%})')
    days=pd.Series(valid.Date.dt.normalize().unique()).sort_values().reset_index(drop=True); rem=days.iloc[max(1,int(len(days)*.45)):].reset_index(drop=True); pooled=[]; reps=[]
    for fi,ch in enumerate(np.array_split(rem,3),1):
        start,end=pd.Timestamp(ch.iloc[0]),pd.Timestamp(ch.iloc[-1]); fit=valid[valid.Date.dt.normalize()<start]; test=valid[(valid.Date.dt.normalize()>=start)&(valid.Date.dt.normalize()<=end)]
        if len(fit)<700 or len(test)<250: continue
        best,trials,cut=_select(fit); m=_fit(fit,best['C']); test=test.copy(); test['candidate_home_prob']=m.predict_proba(_design(test))[:,1]; test['candidate_total_runs']=test.baseline_total_runs; test['fold']=fi; pooled.append(test); reps.append({'fold':fi,'test_start':str(start.date()),'test_end':str(end.date()),'train_rows':len(fit),'test_rows':len(test),'inner_cut':cut,'selected':best,'trials':trials})
    if not pooled: raise SystemExit('No valid platoon walk-forward folds')
    out=pd.concat(pooled).sort_values('Date'); OUT.parent.mkdir(exist_ok=True); out.to_csv(OUT,index=False); report={'policy':'nested_walkforward_confirmed_lineup_real_starter_hand_v1','rows':len(out),'source_rows':source,'qualified_rows':len(valid),'qualified_coverage':coverage,'min_hand_pa':MIN_HAND_PA,'min_qualified_hitters':MIN_QUALIFIED,'features':list(FEATURES),'folds':reps}; REPORT.write_text(json.dumps(report,indent=2)); print('LINEUP_PLATOON_WALKFORWARD',json.dumps(report)); print(f'OK {OUT} rows={len(out)} qualified_coverage={coverage:.1%}')

if __name__=='__main__': main()
