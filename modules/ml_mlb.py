import hashlib
import math
import threading
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .team_utils import normalize_team
from .historical_mlb import prepare_games, team_state, h2h_state, append_game
from .metric_quality import batting_metric, pitching_metric

_MODEL_CACHE = OrderedDict()
_MODEL_CACHE_LOCK = threading.RLock()
_MODEL_CACHE_MAX = 2


def _frame_signature(df, important_columns):
    if df is None or df.empty:
        return (0, 0, "")
    cols = [c for c in important_columns if c in df.columns] or list(df.columns)
    hashed = pd.util.hash_pandas_object(df[cols], index=True).values.tobytes()
    return (int(len(df)), int(len(df.columns)), hashlib.sha256(hashed).hexdigest())


def _cache_key(df_batting, df_pitching, df_games):
    return (
        _frame_signature(df_batting, ('Team','Season','OPS_Index','wRC+','wRC+_Source','wOBA','ISO','BB%','K%')),
        _frame_signature(df_pitching, ('Team','Season','ERA','FIP','xFIP','xFIP_Source','K-BB%','WHIP','GB%','HR/9')),
        _frame_signature(df_games, ('Date','Season','Home','Away','Home_Score','Away_Score','gamePk')),
    )


def _normal_cdf(z):
    return 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))


def _calibrate_sigma(pred, actual, market='spread'):
    pred = np.asarray(pred, dtype=float); actual = np.asarray(actual, dtype=float)
    if len(pred) < 100:
        return float(max(1.0, np.std(actual - pred)))
    base = float(max(1.0, np.std(actual - pred)))
    best_sigma, best_score = base, float('inf')
    if market == 'spread':
        outcomes = [(actual + 1.5 > 0).astype(float), (-actual + 1.5 > 0).astype(float)]
        projections = [pred + 1.5, -pred + 1.5]
    else:
        outcomes, projections = [], []
        for line in (7.5, 8.5, 9.5):
            outcomes += [(actual > line).astype(float), (actual < line).astype(float)]
            projections += [pred - line, line - pred]
    for mult in np.linspace(0.85, 1.75, 37):
        sigma = base * float(mult); scores = []
        for y, margin in zip(outcomes, projections):
            probs = np.array([_normal_cdf(v / sigma) for v in margin], dtype=float)
            scores.append(float(np.mean((probs-y)**2)))
        score = float(np.mean(scores))
        if score < best_score: best_score, best_sigma = score, sigma
    return float(max(1.0, best_sigma))


class PredictorMLMLB:
    """Leak-safe pregame model with optional DuckDB/Parquet feature-store training."""

    def __init__(self):
        self.classifier_family = 'logistic'; self.runs_family = 'gbr'; self.diff_family = 'gbr'
        self.modelo_ganador = self._new_classifier(self.classifier_family)
        self.modelo_carreras = self._new_regressor(self.runs_family)
        self.modelo_handicap = self._new_regressor(self.diff_family)
        self.entrenado = False; self.bat_scale, self.pit_scale = 100.0, 4.10
        self.current_history, self.current_h2h = {}, {}
        self.sigma_runs, self.sigma_diff = 3.5, 4.2; self.prob_shrink = 1.0
        self.loaded_from_cache = False; self.validation_brier = None
        self.validation_runs_mae = None; self.validation_diff_mae = None
        self.training_source = 'csv_chronological'
        self.training_rows = 0

    @staticmethod
    def _new_classifier(family='logistic'):
        if family == 'histgb':
            return HistGradientBoostingClassifier(learning_rate=0.04, max_iter=180, max_leaf_nodes=15,
                min_samples_leaf=30, l2_regularization=2.0, random_state=42)
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.35, max_iter=2000, solver='lbfgs', random_state=42))

    @staticmethod
    def _new_regressor(family='gbr'):
        if family == 'histgb':
            return HistGradientBoostingRegressor(learning_rate=0.04, max_iter=180, max_leaf_nodes=15,
                min_samples_leaf=30, l2_regularization=2.0, random_state=42)
        return GradientBoostingRegressor(n_estimators=140, max_depth=2, learning_rate=0.035, loss='huber', random_state=42)

    @staticmethod
    def _stats_dict(df, col):
        x = df.copy(); x['Team'] = x['Team'].map(normalize_team); x['Season'] = pd.to_numeric(x['Season'], errors='coerce')
        x[col] = pd.to_numeric(x[col], errors='coerce')
        return x.dropna(subset=['Team','Season',col]).set_index(['Team','Season'])[col].to_dict()

    def _feature_row(self, hist, h2h, loc, vis, off_l, off_v, pit_l, pit_v):
        w5l, rf5l, ra5l, rd5l = team_state(hist, loc, 5); w5v, rf5v, ra5v, rd5v = team_state(hist, vis, 5)
        w20l, rf20l, ra20l, rd20l = team_state(hist, loc, 20); w20v, rf20v, ra20v, rd20v = team_state(hist, vis, 20)
        hwin, hrd, hn = h2h_state(h2h, loc, vis, 12)
        return [w5l,w5v,w20l,w20v,rf5l,rf5v,ra5l,ra5v,rd5l,rd5v,rd20l,rd20v,hwin,hrd,min(hn,12)/12.0,
                float(off_l)/max(self.bat_scale,1e-6),float(off_v)/max(self.bat_scale,1e-6),
                float(pit_l)/max(self.pit_scale,1e-6),float(pit_v)/max(self.pit_scale,1e-6),1.0]

    @staticmethod
    def _shrink_probability(raw_prob, alpha):
        return float(np.clip(0.5 + float(alpha)*(float(raw_prob)-0.5), 0.01, 0.99))

    def _restore_cached(self, cached):
        self.modelo_ganador=cached['modelo_ganador']; self.modelo_carreras=cached['modelo_carreras']; self.modelo_handicap=cached['modelo_handicap']
        self.classifier_family=cached.get('classifier_family','logistic'); self.runs_family=cached.get('runs_family','gbr'); self.diff_family=cached.get('diff_family','gbr')
        self.bat_scale=cached['bat_scale']; self.pit_scale=cached['pit_scale']; self.sigma_runs=cached['sigma_runs']; self.sigma_diff=cached['sigma_diff']
        self.prob_shrink=cached['prob_shrink']; self.validation_brier=cached.get('validation_brier'); self.validation_runs_mae=cached.get('validation_runs_mae')
        self.validation_diff_mae=cached.get('validation_diff_mae'); self.current_history=deepcopy(cached['current_history']); self.current_h2h=deepcopy(cached['current_h2h'])
        self.training_source=cached.get('training_source','csv_chronological'); self.training_rows=int(cached.get('training_rows',0)); self.entrenado=True; self.loaded_from_cache=True

    def _training_arrays(self, df_batting, df_pitching, games, bd, pdict):
        """Prefer warehouse features but never exceed the caller's training subset."""
        try:
            from .bigdata_mlb import MLBDataWarehouse, LEGACY_ML_COLUMNS, bootstrap_from_repository
            wh = MLBDataWarehouse()
            if not wh.paths.db.exists():
                bootstrap_from_repository()
                wh = MLBDataWarehouse()
            frame = wh.legacy_ml_training_frame(df_batting, df_pitching, self.bat_scale, self.pit_scale)
            requested = wh._normalize_games(games)
            if not frame.empty and not requested.empty and 'game_key' in frame.columns and 'game_key' in requested.columns:
                requested_keys = set(requested['game_key'].astype(str))
                frame = frame[frame['game_key'].astype(str).isin(requested_keys)].copy()
                frame = frame.sort_values(['Date','game_key']).reset_index(drop=True)
            # A partial match is unsafe: fall back to the caller's chronological CSV subset.
            if (len(frame) == len(requested) and len(frame) >= 1000 and
                    all(c in frame.columns for c in LEGACY_ML_COLUMNS)):
                X = frame[LEGACY_ML_COLUMNS].apply(pd.to_numeric, errors='coerce')
                valid = X.notna().all(axis=1)
                frame = frame.loc[valid].reset_index(drop=True); X = X.loc[valid].to_numpy(float)
                if len(frame) == len(requested):
                    self.training_source = 'duckdb_parquet_feature_store_subset_safe'
                    return X, frame['target_home_win'].to_numpy(int), frame['target_total_runs'].to_numpy(float), frame['target_run_diff'].to_numpy(float)
        except Exception as e:
            print(f'Big Data fallback a CSV: {e}')
        X,yw,yr,yd=[],[],[],[]; hist,hh={},{}
        for _,r in games.iterrows():
            loc,vis=r.Home,r.Away; year=int(r.Season); hs,as_=float(r.Home_Score),float(r.Away_Score); sy=year-1
            ol=float(bd.get((loc,sy),self.bat_scale)); ov=float(bd.get((vis,sy),self.bat_scale)); pl=float(pdict.get((loc,sy),self.pit_scale)); pv_=float(pdict.get((vis,sy),self.pit_scale))
            X.append(self._feature_row(hist,hh,loc,vis,ol,ov,pl,pv_)); yw.append(int(hs>as_)); yr.append(hs+as_); yd.append(hs-as_)
            append_game(hist,hh,loc,vis,hs,as_)
        self.training_source='csv_chronological'
        return np.asarray(X,float),np.asarray(yw),np.asarray(yr),np.asarray(yd)

    def entrenar(self, df_batting, df_pitching, df_games):
        try:
            if df_batting.empty or df_pitching.empty or df_games.empty: return False
            key=_cache_key(df_batting,df_pitching,df_games)
            with _MODEL_CACHE_LOCK:
                cached=_MODEL_CACHE.get(key)
                if cached is not None: _MODEL_CACHE.move_to_end(key); self._restore_cached(cached); return True
            bat_col=batting_metric(df_batting); pit_col=pitching_metric(df_pitching)
            if not bat_col or not pit_col: return False
            games=prepare_games(df_games); bd=self._stats_dict(df_batting,bat_col); pdict=self._stats_dict(df_pitching,pit_col)
            bv=pd.to_numeric(df_batting[bat_col],errors='coerce').dropna(); pv=pd.to_numeric(df_pitching[pit_col],errors='coerce').dropna()
            self.bat_scale=float(bv.median()) if len(bv) else 100.0; self.pit_scale=float(pv.median()) if len(pv) else 4.10
            X,yw,yr,yd=self._training_arrays(df_batting,df_pitching,games,bd,pdict)
            self.training_rows=int(len(X))
            if len(X)<1000: return False
            cut=max(100,int(len(X)*0.80)); cut=max(100,len(X)-50) if cut>=len(X)-50 else cut
            run_candidates={}; diff_candidates={}
            for family in ('gbr','histgb'):
                rm=self._new_regressor(family); rm.fit(X[:cut],yr[:cut]); rp=rm.predict(X[cut:]); run_candidates[family]=(float(np.mean(np.abs(rp-yr[cut:]))),rp)
                dm=self._new_regressor(family); dm.fit(X[:cut],yd[:cut]); dp=dm.predict(X[cut:]); diff_candidates[family]=(float(np.mean(np.abs(dp-yd[cut:]))),dp)
            self.runs_family=min(run_candidates,key=lambda k:run_candidates[k][0]); self.diff_family=min(diff_candidates,key=lambda k:diff_candidates[k][0])
            self.validation_runs_mae=run_candidates[self.runs_family][0]; self.validation_diff_mae=diff_candidates[self.diff_family][0]
            self.sigma_runs=_calibrate_sigma(run_candidates[self.runs_family][1],yr[cut:],'total'); self.sigma_diff=_calibrate_sigma(diff_candidates[self.diff_family][1],yd[cut:],'spread')
            best=None
            for family in ('logistic','histgb'):
                cm=self._new_classifier(family); cm.fit(X[:cut],yw[:cut]); raw=cm.predict_proba(X[cut:])[:,1]
                for alpha in np.linspace(0.35,1.0,66):
                    cal=0.5+alpha*(raw-0.5); brier=float(np.mean((cal-yw[cut:])**2))
                    if best is None or brier<best[0]: best=(brier,family,float(alpha))
            self.validation_brier,self.classifier_family,self.prob_shrink=best
            self.modelo_ganador=self._new_classifier(self.classifier_family); self.modelo_carreras=self._new_regressor(self.runs_family); self.modelo_handicap=self._new_regressor(self.diff_family)
            self.modelo_ganador.fit(X,yw); self.modelo_carreras.fit(X,yr); self.modelo_handicap.fit(X,yd)
            hist,hh={},{}
            for _,r in games.iterrows(): append_game(hist,hh,r.Home,r.Away,float(r.Home_Score),float(r.Away_Score))
            self.current_history,self.current_h2h=hist,hh; self.entrenado=True; self.loaded_from_cache=False
            cache_value={'modelo_ganador':self.modelo_ganador,'modelo_carreras':self.modelo_carreras,'modelo_handicap':self.modelo_handicap,
                'classifier_family':self.classifier_family,'runs_family':self.runs_family,'diff_family':self.diff_family,'bat_scale':self.bat_scale,'pit_scale':self.pit_scale,
                'sigma_runs':self.sigma_runs,'sigma_diff':self.sigma_diff,'prob_shrink':self.prob_shrink,'validation_brier':self.validation_brier,
                'validation_runs_mae':self.validation_runs_mae,'validation_diff_mae':self.validation_diff_mae,'current_history':deepcopy(hist),'current_h2h':deepcopy(hh),
                'training_source':self.training_source,'training_rows':self.training_rows}
            with _MODEL_CACHE_LOCK:
                _MODEL_CACHE[key]=cache_value; _MODEL_CACHE.move_to_end(key)
                while len(_MODEL_CACHE)>_MODEL_CACHE_MAX: _MODEL_CACHE.popitem(last=False)
            return True
        except Exception as e:
            print(f'Error entrenando ML MLB: {e}'); self.entrenado=False; return False

    def actualizar_resultado(self, loc_abbr, vis_abbr, home_score, away_score):
        append_game(self.current_history,self.current_h2h,normalize_team(loc_abbr),normalize_team(vis_abbr),float(home_score),float(away_score))

    @staticmethod
    def _cdf(z): return _normal_cdf(z)

    def predecir_partido(self, loc_abbr, vis_abbr, wrc_loc, wrc_vis, xfip_loc, xfip_vis, pf=None):
        try:
            if not self.entrenado: raise RuntimeError('Modelo no entrenado')
            loc,vis=normalize_team(loc_abbr),normalize_team(vis_abbr)
            f=np.asarray([self._feature_row(self.current_history,self.current_h2h,loc,vis,float(wrc_loc),float(wrc_vis),float(xfip_loc),float(xfip_vis))],float)
            raw_local=float(self.modelo_ganador.predict_proba(f)[0,1]); p_local=self._shrink_probability(raw_local,self.prob_shrink)
            runs=float(self.modelo_carreras.predict(f)[0]); diff=float(self.modelo_handicap.predict(f)[0])
            return {'Probabilidad_Local':round(p_local*100,2),'Probabilidad_Visita':round((1-p_local)*100,2),'Probabilidad_Local_Raw':round(raw_local*100,2),
                'Prob_Shrink':round(self.prob_shrink,3),'Proyeccion_Carreras':round(runs,2),'Proyeccion_Handicap_Local':round(diff,2),
                'Sigma_Carreras':round(self.sigma_runs,3),'Sigma_Handicap':round(self.sigma_diff,3),'Modelo_Desde_Cache':bool(self.loaded_from_cache),
                'Classifier_Family':self.classifier_family,'Runs_Family':self.runs_family,'Diff_Family':self.diff_family,
                'Training_Source':self.training_source,'Training_Rows':self.training_rows,
                'Validation_Brier':None if self.validation_brier is None else round(self.validation_brier,5),
                'Validation_Runs_MAE':None if self.validation_runs_mae is None else round(self.validation_runs_mae,4),
                'Validation_Diff_MAE':None if self.validation_diff_mae is None else round(self.validation_diff_mae,4)}
        except Exception as e:
            print(f'Error en predicción ML: {e}')
            return {'Probabilidad_Local':50.0,'Probabilidad_Visita':50.0,'Probabilidad_Local_Raw':50.0,'Prob_Shrink':1.0,
                    'Proyeccion_Carreras':8.5,'Proyeccion_Handicap_Local':0.0,'Sigma_Carreras':3.5,'Sigma_Handicap':4.2,
                    'Modelo_Desde_Cache':False,'Classifier_Family':'fallback','Runs_Family':'fallback','Diff_Family':'fallback','Training_Source':'fallback','Training_Rows':0}