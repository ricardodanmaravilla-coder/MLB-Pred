import hashlib
import math
import threading
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .team_utils import normalize_team
from .historical_mlb import prepare_games, team_state, h2h_state, append_game

_MODEL_CACHE=OrderedDict();_MODEL_CACHE_LOCK=threading.RLock();_MODEL_CACHE_MAX=2

def _frame_signature(df,important_columns):
    if df is None or df.empty:return (0,0,"")
    cols=[c for c in important_columns if c in df.columns] or list(df.columns);digest=hashlib.sha256(pd.util.hash_pandas_object(df[cols],index=True).values.tobytes()).hexdigest();return (int(len(df)),int(len(df.columns)),digest)
def _cache_key(df_batting,df_pitching,df_games):
    return (_frame_signature(df_batting,('Team','Season','Offense_Index','wRC+','OPS_Index','ISO','K%_Official','BB%_Official')),_frame_signature(df_pitching,('Team','Season','xFIP','FIP','ERA','K-BB%','WHIP','HR/9')),_frame_signature(df_games,('Date','Season','Home','Away','Home_Score','Away_Score')))
def preferred_batting_column(df):
    if df is None or df.empty:return None
    if 'wRC+' in df.columns:
        vals=pd.to_numeric(df['wRC+'],errors='coerce');source=df.get('wRC+_Source',pd.Series('',index=df.index)).astype(str);real=vals[source.str.contains('FANGRAPHS_REAL',case=False,na=False)]
        if real.notna().sum()>=max(20,int(vals.notna().sum()*.5)):return 'wRC+'
    if 'Offense_Index' in df.columns and pd.to_numeric(df['Offense_Index'],errors='coerce').notna().sum()>=20:return 'Offense_Index'
    if 'OPS_Index' in df.columns:return 'OPS_Index'
    return 'wRC+' if 'wRC+' in df.columns else None
def preferred_pitching_column(df):
    if df is None or df.empty:return None
    if 'xFIP' in df.columns:
        vals=pd.to_numeric(df['xFIP'],errors='coerce');source=df.get('xFIP_Source',pd.Series('',index=df.index)).astype(str);real=vals[source.str.contains('FANGRAPHS_REAL',case=False,na=False)]
        if real.notna().sum()>=max(20,int(vals.notna().sum()*.5)):return 'xFIP'
    if 'FIP' in df.columns and pd.to_numeric(df['FIP'],errors='coerce').notna().sum()>=20:return 'FIP'
    return 'ERA' if 'ERA' in df.columns else ('xFIP' if 'xFIP' in df.columns else None)
def _normal_cdf(z):return .5*(1.+math.erf(float(z)/math.sqrt(2.)))
def _calibrate_sigma(pred,actual,market='spread'):
    pred,actual=np.asarray(pred,float),np.asarray(actual,float)
    if len(pred)<100:return float(max(1.,np.std(actual-pred)))
    base=float(max(1.,np.std(actual-pred)));best_sigma,best_score=base,float('inf')
    if market=='spread':outcomes=[(actual+1.5>0).astype(float),(-actual+1.5>0).astype(float)];projections=[pred+1.5,-pred+1.5]
    else:
        outcomes=[];projections=[]
        for line in (7.5,8.5,9.5):outcomes += [(actual>line).astype(float),(actual<line).astype(float)];projections += [pred-line,line-pred]
    for mult in np.linspace(.85,1.75,37):
        sigma=base*float(mult);scores=[]
        for y,margin in zip(outcomes,projections):probs=np.array([_normal_cdf(v/sigma) for v in margin]);scores.append(float(np.mean((probs-y)**2)))
        score=float(np.mean(scores))
        if score<best_score:best_score,best_sigma=score,sigma
    return float(max(1.,best_sigma))

class PredictorMLMLB:
    def __init__(self):
        self.modelo_ganador=self._new_classifier();self.modelo_carreras=GradientBoostingRegressor(n_estimators=140,max_depth=2,learning_rate=.035,loss='huber',random_state=42);self.modelo_handicap=GradientBoostingRegressor(n_estimators=140,max_depth=2,learning_rate=.035,loss='huber',random_state=42);self.entrenado=False;self.bat_scale=100.;self.pit_scale=4.10;self.current_history={};self.current_h2h={};self.sigma_runs=3.5;self.sigma_diff=4.2;self.prob_shrink=1.;self.loaded_from_cache=False;self.batting_metric=None;self.pitching_metric=None
    @staticmethod
    def _new_classifier():return make_pipeline(StandardScaler(),LogisticRegression(C=.35,max_iter=2000,solver='lbfgs',random_state=42))
    @staticmethod
    def _stats_dict(df,col):
        x=df.copy();x['Team']=x['Team'].map(normalize_team);x['Season']=pd.to_numeric(x['Season'],errors='coerce');x[col]=pd.to_numeric(x[col],errors='coerce');return x.dropna(subset=['Team','Season',col]).set_index(['Team','Season'])[col].to_dict()
    def _feature_row(self,hist,h2h,loc,vis,off_l,off_v,pit_l,pit_v):
        w5l,rf5l,ra5l,rd5l=team_state(hist,loc,5);w5v,rf5v,ra5v,rd5v=team_state(hist,vis,5);w20l,rf20l,ra20l,rd20l=team_state(hist,loc,20);w20v,rf20v,ra20v,rd20v=team_state(hist,vis,20);hwin,hrd,hn=h2h_state(h2h,loc,vis,12)
        return [w5l,w5v,w20l,w20v,rf5l,rf5v,ra5l,ra5v,rd5l,rd5v,rd20l,rd20v,hwin,hrd,min(hn,12)/12.,float(off_l)/max(self.bat_scale,1e-6),float(off_v)/max(self.bat_scale,1e-6),float(pit_l)/max(self.pit_scale,1e-6),float(pit_v)/max(self.pit_scale,1e-6),1.]
    @staticmethod
    def _shrink_probability(raw_prob,alpha):return float(np.clip(.5+float(alpha)*(float(raw_prob)-.5),.01,.99))
    def _restore_cached(self,cached):
        for k in ('modelo_ganador','modelo_carreras','modelo_handicap','bat_scale','pit_scale','sigma_runs','sigma_diff','prob_shrink','batting_metric','pitching_metric'):setattr(self,k,cached[k])
        self.current_history=deepcopy(cached['current_history']);self.current_h2h=deepcopy(cached['current_h2h']);self.entrenado=True;self.loaded_from_cache=True
    def entrenar(self,df_batting,df_pitching,df_games):
        try:
            if df_batting.empty or df_pitching.empty or df_games.empty:return False
            key=_cache_key(df_batting,df_pitching,df_games)
            with _MODEL_CACHE_LOCK:
                cached=_MODEL_CACHE.get(key)
                if cached is not None:_MODEL_CACHE.move_to_end(key);self._restore_cached(cached);return True
            bat_col=preferred_batting_column(df_batting);pit_col=preferred_pitching_column(df_pitching)
            if not bat_col or not pit_col:return False
            self.batting_metric,self.pitching_metric=bat_col,pit_col;games=prepare_games(df_games);bd=self._stats_dict(df_batting,bat_col);pdict=self._stats_dict(df_pitching,pit_col);bv=pd.to_numeric(df_batting[bat_col],errors='coerce').dropna();pvals=pd.to_numeric(df_pitching[pit_col],errors='coerce').dropna();self.bat_scale=float(bv.median()) if len(bv) else 100.;self.pit_scale=float(pvals.median()) if len(pvals) else 4.10
            X=[];yw=[];yr=[];yd=[];hist={};hh={}
            for _,r in games.iterrows():
                loc,vis=r.Home,r.Away;year=int(r.Season);hs,as_=float(r.Home_Score),float(r.Away_Score);sy=year-1;ol=float(bd.get((loc,sy),self.bat_scale));ov=float(bd.get((vis,sy),self.bat_scale));pl=float(pdict.get((loc,sy),self.pit_scale));pv=float(pdict.get((vis,sy),self.pit_scale));X.append(self._feature_row(hist,hh,loc,vis,ol,ov,pl,pv));yw.append(int(hs>as_));yr.append(hs+as_);yd.append(hs-as_);append_game(hist,hh,loc,vis,hs,as_)
            if len(X)<1000:return False
            X=np.asarray(X,float);yw=np.asarray(yw);yr=np.asarray(yr);yd=np.asarray(yd);cut=max(100,int(len(X)*.80));cut=max(100,len(X)-50) if cut>=len(X)-50 else cut;self.modelo_carreras.fit(X[:cut],yr[:cut]);self.modelo_handicap.fit(X[:cut],yd[:cut]);self.sigma_runs=_calibrate_sigma(self.modelo_carreras.predict(X[cut:]),yr[cut:],'total');self.sigma_diff=_calibrate_sigma(self.modelo_handicap.predict(X[cut:]),yd[cut:],'spread');temp=self._new_classifier();temp.fit(X[:cut],yw[:cut]);raw=temp.predict_proba(X[cut:])[:,1];best_alpha,best_brier=1.,float('inf')
            for alpha in np.linspace(.35,1.,66):
                score=float(np.mean((.5+alpha*(raw-.5)-yw[cut:])**2))
                if score<best_brier:best_brier,best_alpha=score,float(alpha)
            self.prob_shrink=best_alpha;self.modelo_ganador=self._new_classifier();self.modelo_ganador.fit(X,yw);self.modelo_carreras.fit(X,yr);self.modelo_handicap.fit(X,yd);self.current_history,self.current_h2h=hist,hh;self.entrenado=True;self.loaded_from_cache=False;cache={'modelo_ganador':self.modelo_ganador,'modelo_carreras':self.modelo_carreras,'modelo_handicap':self.modelo_handicap,'bat_scale':self.bat_scale,'pit_scale':self.pit_scale,'sigma_runs':self.sigma_runs,'sigma_diff':self.sigma_diff,'prob_shrink':self.prob_shrink,'batting_metric':self.batting_metric,'pitching_metric':self.pitching_metric,'current_history':deepcopy(hist),'current_h2h':deepcopy(hh)}
            with _MODEL_CACHE_LOCK:
                _MODEL_CACHE[key]=cache;_MODEL_CACHE.move_to_end(key)
                while len(_MODEL_CACHE)>_MODEL_CACHE_MAX:_MODEL_CACHE.popitem(last=False)
            return True
        except Exception as e:print(f'Error entrenando ML MLB: {e}');self.entrenado=False;return False
    def actualizar_resultado(self,loc_abbr,vis_abbr,home_score,away_score):append_game(self.current_history,self.current_h2h,normalize_team(loc_abbr),normalize_team(vis_abbr),float(home_score),float(away_score))
    def predecir_partido(self,loc_abbr,vis_abbr,wrc_loc,wrc_vis,xfip_loc,xfip_vis,pf=None):
        try:
            if not self.entrenado:raise RuntimeError('Modelo no entrenado')
            loc,vis=normalize_team(loc_abbr),normalize_team(vis_abbr);f=np.asarray([self._feature_row(self.current_history,self.current_h2h,loc,vis,float(wrc_loc),float(wrc_vis),float(xfip_loc),float(xfip_vis))],float);raw=float(self.modelo_ganador.predict_proba(f)[0,1]);pl=self._shrink_probability(raw,self.prob_shrink);runs=float(self.modelo_carreras.predict(f)[0]);diff=float(self.modelo_handicap.predict(f)[0]);return {'Probabilidad_Local':round(pl*100,2),'Probabilidad_Visita':round((1-pl)*100,2),'Probabilidad_Local_Raw':round(raw*100,2),'Prob_Shrink':round(self.prob_shrink,3),'Proyeccion_Carreras':round(runs,2),'Proyeccion_Handicap_Local':round(diff,2),'Sigma_Carreras':round(self.sigma_runs,3),'Sigma_Handicap':round(self.sigma_diff,3),'Modelo_Desde_Cache':bool(self.loaded_from_cache),'Metrica_Bateo':self.batting_metric,'Metrica_Pitcheo':self.pitching_metric}
        except Exception as e:print(f'Error en predicción ML: {e}');return {'Probabilidad_Local':50.,'Probabilidad_Visita':50.,'Probabilidad_Local_Raw':50.,'Prob_Shrink':1.,'Proyeccion_Carreras':8.5,'Proyeccion_Handicap_Local':0.,'Sigma_Carreras':3.5,'Sigma_Handicap':4.2,'Modelo_Desde_Cache':False,'Metrica_Bateo':None,'Metrica_Pitcheo':None}
