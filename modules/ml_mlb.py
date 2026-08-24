import math
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .team_utils import normalize_team
from .historical_mlb import prepare_games, team_state, h2h_state, append_game


class PredictorMLMLB:
    """Pre-game MLB model: actual historical results + prior-season offense/defense."""
    def __init__(self):
        self.modelo_ganador = make_pipeline(StandardScaler(), LogisticRegression(C=0.35, max_iter=2000, solver='lbfgs', random_state=42))
        self.modelo_carreras = GradientBoostingRegressor(n_estimators=140, max_depth=2, learning_rate=0.035, loss='huber', random_state=42)
        self.modelo_handicap = GradientBoostingRegressor(n_estimators=140, max_depth=2, learning_rate=0.035, loss='huber', random_state=42)
        self.entrenado = False
        self.bat_scale, self.pit_scale = 100.0, 4.10
        self.current_history, self.current_h2h = {}, {}
        self.sigma_runs, self.sigma_diff = 3.5, 4.2

    @staticmethod
    def _stats_dict(df, col):
        x=df.copy(); x['Team']=x['Team'].map(normalize_team); x['Season']=pd.to_numeric(x['Season'],errors='coerce'); x[col]=pd.to_numeric(x[col],errors='coerce')
        return x.dropna(subset=['Team','Season',col]).set_index(['Team','Season'])[col].to_dict()

    def _feature_row(self, hist, h2h, loc, vis, off_l, off_v, pit_l, pit_v):
        w5l,rf5l,ra5l,rd5l=team_state(hist,loc,5); w5v,rf5v,ra5v,rd5v=team_state(hist,vis,5)
        w20l,rf20l,ra20l,rd20l=team_state(hist,loc,20); w20v,rf20v,ra20v,rd20v=team_state(hist,vis,20)
        hwin, hrd, hn = h2h_state(h2h,loc,vis,12)
        return [w5l,w5v,w20l,w20v,rf5l,rf5v,ra5l,ra5v,rd5l,rd5v,rd20l,rd20v,hwin,hrd,min(hn,12)/12.0,
                float(off_l)/max(self.bat_scale,1e-6),float(off_v)/max(self.bat_scale,1e-6),float(pit_l)/max(self.pit_scale,1e-6),float(pit_v)/max(self.pit_scale,1e-6),1.0]

    def entrenar(self, df_batting, df_pitching, df_games):
        try:
            if df_batting.empty or df_pitching.empty or df_games.empty: return False
            bat_col='OPS_Index' if 'OPS_Index' in df_batting.columns else 'wRC+'
            pit_col='ERA' if 'ERA' in df_pitching.columns else 'xFIP'
            if bat_col not in df_batting or pit_col not in df_pitching: return False
            games=prepare_games(df_games)
            bd=self._stats_dict(df_batting,bat_col); pdict=self._stats_dict(df_pitching,pit_col)
            bv=pd.to_numeric(df_batting[bat_col],errors='coerce').dropna(); pv=pd.to_numeric(df_pitching[pit_col],errors='coerce').dropna()
            self.bat_scale=float(bv.median()) if len(bv) else 100.; self.pit_scale=float(pv.median()) if len(pv) else 4.10
            X=[]; yw=[]; yr=[]; yd=[]; hist={}; hh={}
            for _,r in games.iterrows():
                loc,vis=r.Home,r.Away; year=int(r.Season); hs,as_=float(r.Home_Score),float(r.Away_Score); sy=year-1
                ol=float(bd.get((loc,sy),self.bat_scale)); ov=float(bd.get((vis,sy),self.bat_scale)); pl=float(pdict.get((loc,sy),self.pit_scale)); pv_=float(pdict.get((vis,sy),self.pit_scale))
                X.append(self._feature_row(hist,hh,loc,vis,ol,ov,pl,pv_)); yw.append(int(hs>as_)); yr.append(hs+as_); yd.append(hs-as_)
                append_game(hist,hh,loc,vis,hs,as_)
            if len(X)<1000: return False
            X=np.asarray(X,float); yw=np.asarray(yw); yr=np.asarray(yr); yd=np.asarray(yd)
            cut=max(100,int(len(X)*.80))
            # Residual calibration is chronological, not in-sample.
            self.modelo_carreras.fit(X[:cut],yr[:cut]); self.modelo_handicap.fit(X[:cut],yd[:cut])
            self.sigma_runs=float(max(1.0,np.std(yr[cut:]-self.modelo_carreras.predict(X[cut:]))))
            self.sigma_diff=float(max(1.0,np.std(yd[cut:]-self.modelo_handicap.predict(X[cut:]))))
            self.modelo_ganador.fit(X,yw); self.modelo_carreras.fit(X,yr); self.modelo_handicap.fit(X,yd)
            self.current_history,self.current_h2h=hist,hh; self.entrenado=True; return True
        except Exception as e:
            print(f'Error entrenando ML MLB: {e}'); self.entrenado=False; return False

    @staticmethod
    def _cdf(z): return 0.5*(1.0+math.erf(z/math.sqrt(2.0)))

    def predecir_partido(self, loc_abbr, vis_abbr, wrc_loc, wrc_vis, xfip_loc, xfip_vis, pf=None):
        try:
            if not self.entrenado: raise RuntimeError('Modelo no entrenado')
            loc,vis=normalize_team(loc_abbr),normalize_team(vis_abbr)
            f=np.asarray([self._feature_row(self.current_history,self.current_h2h,loc,vis,float(wrc_loc),float(wrc_vis),float(xfip_loc),float(xfip_vis))],float)
            pr=self.modelo_ganador.predict_proba(f)[0]; runs=float(self.modelo_carreras.predict(f)[0]); diff=float(self.modelo_handicap.predict(f)[0])
            return {'Probabilidad_Local':round(float(pr[1])*100,2),'Probabilidad_Visita':round(float(pr[0])*100,2),'Proyeccion_Carreras':round(runs,2),'Proyeccion_Handicap_Local':round(diff,2),'Sigma_Carreras':round(self.sigma_runs,3),'Sigma_Handicap':round(self.sigma_diff,3)}
        except Exception as e:
            print(f'Error en predicción ML: {e}'); return {'Probabilidad_Local':50.,'Probabilidad_Visita':50.,'Proyeccion_Carreras':8.5,'Proyeccion_Handicap_Local':0.,'Sigma_Carreras':3.5,'Sigma_Handicap':4.2}
