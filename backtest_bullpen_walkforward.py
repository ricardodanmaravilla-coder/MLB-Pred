"""Walk-forward validation of real pregame bullpen availability.

Leak-safe: every state uses only appearances strictly before game date. This revision
focuses on WHO is unavailable: season-to-date reliever quality/usage identifies high
leverage arms, then measures their recent workload and availability.
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
OUT=Path('artifacts/bullpen_walkforward_predictions.csv'); REPORT=Path('artifacts/bullpen_walkforward_report.json')
C_GRID=(0.005,0.01,0.03,0.10)
FEATURE_SETS={
 'leverage_unavailable':('bp_top2_load_diff','bp_top3_load_diff'),
 'leverage_recent':('bp_top2_pitches2_diff','bp_top3_pitches3_diff'),
 'leverage_b2b':('bp_top3_b2b_diff','bp_top3_heavy_diff'),
 'leverage_quality_risk':('bp_quality_tired_diff','bp_top2_load_diff'),
 'leverage_core':('bp_top2_load_diff','bp_top3_load_diff','bp_top3_b2b_diff','bp_top3_heavy_diff'),
 'availability_plus_leverage':('bp_pitches1_diff','bp_pitches3_diff','bp_heavy2_diff','bp_top2_load_diff','bp_top3_b2b_diff'),
}
def _date_boundary(games,frac=.70):
 d=pd.Series(games.Date.dt.normalize().dropna().unique()).sort_values().reset_index(drop=True); return pd.Timestamp(d.iloc[max(1,min(len(d)-1,int(len(d)*frac)))])
def _maps(df,m):
 x=df.copy(); x['Team']=x.Team.map(normalize_team); x['Season']=pd.to_numeric(x.Season,errors='coerce'); x[m]=pd.to_numeric(x[m],errors='coerce'); return x.dropna(subset=['Team','Season',m]).set_index(['Team','Season'])[m].to_dict()
def _logit(p): p=np.clip(np.asarray(p,float),1e-5,1-1e-5); return np.log(p/(1-p))
def _design(df,cols): return np.hstack([_logit(df.baseline_home_prob).reshape(-1,1),df[list(cols)].to_numpy(float)])
def _brier(y,p): return float(np.mean((np.asarray(p)-np.asarray(y))**2))
def _ll(y,p): p=np.clip(np.asarray(p),1e-9,1-1e-9); y=np.asarray(y); return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
def _ece(y,p):
 y=np.asarray(y); p=np.asarray(p); z=0.; e=np.linspace(0,1,11)
 for i in range(10):
  m=(p>=e[i])&((p<e[i+1]) if i<9 else (p<=e[i+1])); z+=(m.sum()/len(y))*abs(float(p[m].mean()-y[m].mean())) if m.any() else 0
 return float(z)
def _fit(df,cols,c):
 m=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=2000,solver='lbfgs',random_state=42)); m.fit(_design(df,cols),df.actual_home_win.astype(int)); return m
def _prep_usage(path):
 u=pd.read_csv(path); u['Date']=pd.to_datetime(u.Date,errors='coerce'); u['Team']=u.Team.map(normalize_team)
 for c in ('Pitches','BF','ER','BB','SO','HR','IP'): u[c]=pd.to_numeric(u[c],errors='coerce')
 return u.dropna(subset=['Date','Team','PitcherID']).sort_values(['Team','Date','GameID'])
def _reliever_table(prior):
 if prior.empty:return pd.DataFrame()
 g=prior.groupby('PitcherID',as_index=False)[['Pitches','BF','BB','SO','HR','ER','IP']].sum(min_count=1); g=g[(g.BF>=25)&(g.IP>=8)].copy()
 if g.empty:return g
 g['kbb']=(g.SO.fillna(0)-g.BB.fillna(0))/g.BF.replace(0,np.nan); g['hr9']=9*g.HR.fillna(0)/g.IP.replace(0,np.nan); g['era']=9*g.ER.fillna(0)/g.IP.replace(0,np.nan)
 # Leverage proxy is pregame-only: quality plus how often the manager has used the arm.
 q=1.8*g.kbb-.055*g.hr9-.015*g.era; qmed=q.median(); qs=(q.quantile(.75)-q.quantile(.25)) or q.std() or 1.; g['quality']=((q-qmed)/(2.5*qs)).clip(-.5,.5)
 ipmed=g.IP.median() or 1.; g['role_score']=g.quality+0.18*np.log1p(g.IP/ipmed)+0.08*np.log1p(g.Pitches.fillna(0)/max(float(g.Pitches.median() or 1),1)); return g.sort_values('role_score',ascending=False)
def _team_state(tu,date):
 prior=tu[tu.Date<date]; keys=('pitches1','pitches3','heavy2','quality_tired','top2_load','top3_load','top2_pitches2','top3_pitches3','top3_b2b','top3_heavy')
 if prior.empty:return {k:0. for k in keys}
 d=pd.Timestamp(date).normalize(); r1=prior[prior.Date>=d-pd.Timedelta(days=1)]; r2=prior[prior.Date>=d-pd.Timedelta(days=2)]; r3=prior[prior.Date>=d-pd.Timedelta(days=3)]
 tab=_reliever_table(prior); top2=set(tab.head(2).PitcherID.astype(int)); top3=set(tab.head(3).PitcherID.astype(int)); q=dict(zip(tab.PitcherID.astype(int),tab.quality.fillna(0)))
 def sump(x,ids=None):
  if ids is not None:x=x[x.PitcherID.astype(int).isin(ids)]
  return float(x.Pitches.fillna(0).sum())
 by2=r2.groupby('PitcherID').Pitches.sum(min_count=1) if len(r2) else pd.Series(dtype=float); dates=r2.groupby('PitcherID').Date.nunique() if len(r2) else pd.Series(dtype=float)
 tired=sum(max(0.,q.get(int(pid),0.))*min(float(v or 0)/40.,1.5) for pid,v in by2.fillna(0).items() if v>=20)
 def load(ids):
  vals=[]
  for pid in ids:
   p1=sump(r1,{pid}); p2=sump(r2,{pid}); p3=sump(r3,{pid}); vals.append(min(1.5,p1/25.)+.65*min(1.5,p2/40.)+.35*min(1.5,p3/60.))
  return float(sum(vals))
 return {'pitches1':sump(r1)/100.,'pitches3':sump(r3)/220.,'heavy2':float((by2.fillna(0)>=30).sum())/3.,'quality_tired':float(tired),'top2_load':load(top2),'top3_load':load(top3),'top2_pitches2':sump(r2,top2)/70.,'top3_pitches3':sump(r3,top3)/130.,'top3_b2b':float(sum(dates.get(pid,0)>=2 for pid in top3))/3.,'top3_heavy':float(sum(by2.get(pid,0)>=30 for pid in top3))/3.}
def _build_states(u,games):
 teams={t:x.copy() for t,x in u.groupby('Team')}; cache={}; rows=[]
 for _,r in games.iterrows():
  d=pd.Timestamp(r.Date).normalize(); h,a=normalize_team(r.Home),normalize_team(r.Away)
  def st(t):
   k=(t,d)
   if k not in cache:cache[k]=_team_state(teams.get(t,pd.DataFrame(columns=u.columns)),d)
   return cache[k]
  hs,aa=st(h),st(a); z={f'bp_{k}_diff':hs[k]-aa[k] for k in hs}; z['bullpen_usage_coverage']=float(h in teams and a in teams); rows.append(z)
 return pd.DataFrame(rows,index=games.index)
def _select(train):
 days=pd.Series(train.Date.dt.normalize().unique()).sort_values().reset_index(drop=True); cut=pd.Timestamp(days.iloc[max(1,min(len(days)-1,int(len(days)*.75)))]); tr=train[train.Date.dt.normalize()<cut]; va=train[train.Date.dt.normalize()>=cut]
 y=va.actual_home_win.astype(int); base=va.baseline_home_prob; bb,bl,be=_brier(y,base),_ll(y,base),_ece(y,base); trials=[]
 for name,cols in FEATURE_SETS.items():
  for c in C_GRID:
   m=_fit(tr,cols,c); p=m.predict_proba(_design(va,cols))[:,1]; b,l,e=_brier(y,p),_ll(y,p),_ece(y,p); score=b+.15*l+.10*max(0,e-be)+.25*max(0,b-bb); trials.append({'name':name,'C':c,'features':list(cols),'score':score,'brier':b,'log_loss':l,'ece':e})
 trials.sort(key=lambda x:(x['score'],x['brier'],x['log_loss'],len(x['features']))); return trials[0],trials[:5],str(cut.date())
def main():
 bat=pd.read_csv('data/mlb_batting.csv'); pit=pd.read_csv('data/mlb_pitching.csv'); games=prepare_games(pd.read_csv('data/mlb_games.csv')); d1=_date_boundary(games); train=games[games.Date.dt.normalize()<d1]; later=games[games.Date.dt.normalize()>=d1]
 model=PredictorMLMLB(); assert model.entrenar(bat,pit,train); bc,pc=batting_metric(bat),pitching_metric(pit); bd,pdct=_maps(bat,bc),_maps(pit,pc); bmed=float(pd.to_numeric(bat[bc],errors='coerce').median()); pmed=float(pd.to_numeric(pit[pc],errors='coerce').median()); rows=[]
 for day,dg in later.groupby(later.Date.dt.normalize(),sort=True):
  pending=[]
  for idx,r in dg.iterrows():
   h,a=normalize_team(r.Home),normalize_team(r.Away); sy=int(r.Season)-1; pred=model.predecir_partido(h,a,float(bd.get((h,sy),bmed)),float(bd.get((a,sy),bmed)),float(pdct.get((h,sy),pmed)),float(pdct.get((a,sy),pmed)),game_date=r.Date); hs,aw=float(r.Home_Score),float(r.Away_Score); rows.append({'idx':idx,'Date':r.Date,'actual_home_win':int(hs>aw),'actual_total_runs':hs+aw,'baseline_home_prob':float(pred['Probabilidad_Local'])/100.,'baseline_total_runs':float(pred['Proyeccion_Carreras'])}); pending.append((h,a,hs,aw,r.Date))
  for x in pending:model.actualizar_resultado(x[0],x[1],x[2],x[3],game_date=x[4])
 frame=pd.DataFrame(rows).set_index('idx'); u=_prep_usage('data/mlb_bullpen_usage_history.csv'); frame=frame.join(_build_states(u,later.loc[frame.index])).sort_values('Date').reset_index(drop=True); frame.Date=pd.to_datetime(frame.Date); days=pd.Series(frame.Date.dt.normalize().unique()).sort_values().reset_index(drop=True); rem=days.iloc[max(1,int(len(days)*.45)):].reset_index(drop=True); pooled=[]; reps=[]
 for fi,ch in enumerate(np.array_split(rem,3),1):
  start,end=pd.Timestamp(ch.iloc[0]),pd.Timestamp(ch.iloc[-1]); fit=frame[frame.Date.dt.normalize()<start]; test=frame[(frame.Date.dt.normalize()>=start)&(frame.Date.dt.normalize()<=end)]
  if len(fit)<1000 or len(test)<300:continue
  best,top,cut=_select(fit); m=_fit(fit,best['features'],best['C']); test=test.copy(); test['candidate_home_prob']=m.predict_proba(_design(test,best['features']))[:,1]; test['candidate_total_runs']=test.baseline_total_runs; test['walkforward_fold']=fi; pooled.append(test); reps.append({'fold':fi,'test_start':str(start.date()),'test_end':str(end.date()),'train_rows':len(fit),'test_rows':len(test),'inner_cut':cut,'selected':best,'top5':top})
 out=pd.concat(pooled).sort_values('Date'); OUT.parent.mkdir(exist_ok=True); out.to_csv(OUT,index=False); report={'policy':'nested_walkforward_high_leverage_bullpen_v2','rows':len(out),'folds':reps,'usage_rows':len(u),'usage_games':int(u.GameID.nunique()),'coverage':float(out.bullpen_usage_coverage.mean())}; REPORT.write_text(json.dumps(report,indent=2)); print('BULLPEN_WALKFORWARD',json.dumps(report)); print(f'OK {OUT} rows={len(out)} coverage={report["coverage"]:.1%}')
if __name__=='__main__':main()
