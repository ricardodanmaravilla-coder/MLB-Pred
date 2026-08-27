import hashlib
import math
import threading
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .historical_mlb import append_game, h2h_state, prepare_games, team_state
from .metric_quality import batting_metric, pitching_metric
from .signal_features import build_advanced_signal_frame, build_live_signal_row, coverage_report
from .team_utils import normalize_team

_MODEL_CACHE = OrderedDict()
_MODEL_CACHE_LOCK = threading.RLock()
_MODEL_CACHE_MAX = 2
_MUTABLE_CACHE_FIELDS = {
    'current_history', 'current_h2h', 'signal_coverage', 'signal_batting',
    'signal_pitching', 'last_game_dates'
}


def _frame_signature(df, important_columns):
    if df is None or df.empty:
        return (0, 0, '')
    cols = [c for c in important_columns if c in df.columns] or list(df.columns)
    hashed = pd.util.hash_pandas_object(df[cols], index=True).values.tobytes()
    return (int(len(df)), int(len(df.columns)), hashlib.sha256(hashed).hexdigest())


def _cache_key(df_batting, df_pitching, df_games):
    return (
        _frame_signature(df_batting, (
            'Team','Season','OPS_Index','wRC+','wRC+_Source','wOBA','ISO','BB%','K%',
            'EV','HardHit%','Barrel%','OPS_vs_L','OPS_vs_R','OBP_vs_L','OBP_vs_R','SLG_vs_L','SLG_vs_R'
        )),
        _frame_signature(df_pitching, (
            'Team','Season','ERA','FIP','xFIP','SIERA','xFIP_Source','K-BB%','WHIP','GB%','HR/9'
        )),
        _frame_signature(df_games, ('Date','Season','Home','Away','Home_Score','Away_Score','gamePk')),
    )


def _normal_cdf(z):
    return 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))


def _calibrate_sigma(pred, actual, market='spread'):
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
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
        sigma = base * float(mult)
        scores = []
        for y, margin in zip(outcomes, projections):
            probs = np.array([_normal_cdf(v / sigma) for v in margin], dtype=float)
            scores.append(float(np.mean((probs - y) ** 2)))
        score = float(np.mean(scores))
        if score < best_score:
            best_score, best_sigma = score, sigma
    return float(max(1.0, best_sigma))


def _date_safe_cut(dates, ratio=0.80, min_train=100, min_validation=50):
    d = pd.to_datetime(pd.Series(dates), errors='coerce').dt.normalize()
    n = len(d)
    if n < min_train + min_validation:
        return max(1, min(n - 1, int(n * ratio)))
    target = max(int(min_train), min(n - int(min_validation), int(n * ratio)))
    if target <= 0 or target >= n:
        return target
    boundary = d.iloc[target - 1]
    cut = target
    while cut < n and d.iloc[cut] == boundary:
        cut += 1
    if n - cut < min_validation:
        cut = target
        next_day = d.iloc[target]
        while cut > min_train and d.iloc[cut - 1] == next_day:
            cut -= 1
    return int(max(min_train, min(n - min_validation, cut)))


def _as_of_date(value=None):
    if value is None:
        return pd.Timestamp.now(tz='America/New_York').tz_localize(None).normalize()
    try:
        d = pd.Timestamp(value)
        if d.tzinfo is not None:
            d = d.tz_convert('America/New_York').tz_localize(None)
        return d.normalize()
    except Exception:
        return pd.Timestamp.now(tz='America/New_York').tz_localize(None).normalize()


class PredictorMLMLB:
    """Leak-safe MLB model with validation-gated advanced pregame signals."""

    def __init__(self):
        self.classifier_family = 'logistic'
        self.runs_family = 'gbr'
        self.diff_family = 'gbr'
        self.modelo_ganador = self._new_classifier()
        self.modelo_carreras = self._new_regressor()
        self.modelo_handicap = self._new_regressor()
        self.entrenado = False
        self.bat_scale, self.pit_scale = 100.0, 4.10
        self.current_history, self.current_h2h = {}, {}
        self.sigma_runs, self.sigma_diff = 3.5, 4.2
        self.prob_shrink = 1.0
        self.loaded_from_cache = False
        self.validation_brier = None
        self.validation_runs_mae = None
        self.validation_diff_mae = None
        self.validation_cut_date = None
        self.training_source = 'csv_chronological_daily_safe'
        self.training_rows = 0
        self.signal_set = 'baseline20'
        self.signal_gain = None
        self.signal_coverage = {}
        self.signal_batting = pd.DataFrame()
        self.signal_pitching = pd.DataFrame()
        self.signal_season = None
        self.last_game_dates = {}
        self.history_as_of = None

    @staticmethod
    def _new_classifier(family='logistic'):
        if family == 'histgb':
            return HistGradientBoostingClassifier(
                learning_rate=0.04, max_iter=180, max_leaf_nodes=15,
                min_samples_leaf=30, l2_regularization=2.0, random_state=42
            )
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.35, max_iter=2000, solver='lbfgs', random_state=42),
        )

    @staticmethod
    def _new_regressor(family='gbr'):
        if family == 'histgb':
            return HistGradientBoostingRegressor(
                learning_rate=0.04, max_iter=180, max_leaf_nodes=15,
                min_samples_leaf=30, l2_regularization=2.0, random_state=42
            )
        return GradientBoostingRegressor(
            n_estimators=140, max_depth=2, learning_rate=0.035,
            loss='huber', random_state=42
        )

    @staticmethod
    def _stats_dict(df, col):
        x = df.copy()
        x['Team'] = x['Team'].map(normalize_team)
        x['Season'] = pd.to_numeric(x['Season'], errors='coerce')
        x[col] = pd.to_numeric(x[col], errors='coerce')
        return x.dropna(subset=['Team','Season',col]).set_index(['Team','Season'])[col].to_dict()

    def _feature_row(self, hist, h2h, loc, vis, off_l, off_v, pit_l, pit_v):
        w5l, rf5l, ra5l, rd5l = team_state(hist, loc, 5)
        w5v, rf5v, ra5v, rd5v = team_state(hist, vis, 5)
        w20l, _, _, rd20l = team_state(hist, loc, 20)
        w20v, _, _, rd20v = team_state(hist, vis, 20)
        hwin, hrd, hn = h2h_state(h2h, loc, vis, 12)
        return [
            w5l,w5v,w20l,w20v,rf5l,rf5v,ra5l,ra5v,rd5l,rd5v,rd20l,rd20v,
            hwin,hrd,min(hn,12)/12.0,
            float(off_l)/max(self.bat_scale,1e-6), float(off_v)/max(self.bat_scale,1e-6),
            float(pit_l)/max(self.pit_scale,1e-6), float(pit_v)/max(self.pit_scale,1e-6), 1.0,
        ]

    @staticmethod
    def _shrink_probability(raw_prob, alpha):
        return float(np.clip(0.5 + float(alpha) * (float(raw_prob) - 0.5), 0.01, 0.99))

    def _restore_cached(self, cached):
        for key, value in cached.items():
            setattr(self, key, deepcopy(value) if key in _MUTABLE_CACHE_FIELDS else value)
        self.entrenado = True
        self.loaded_from_cache = True

    def _training_arrays(self, df_batting, df_pitching, games, bd, pdict):
        try:
            from .bigdata_mlb import MLBDataWarehouse, LEGACY_ML_COLUMNS, bootstrap_from_repository
            wh = MLBDataWarehouse()
            if not wh.paths.db.exists():
                bootstrap_from_repository()
                wh = MLBDataWarehouse()
            frame = wh.legacy_ml_training_frame(df_batting, df_pitching, self.bat_scale, self.pit_scale)
            requested = wh._normalize_games(games)
            if not frame.empty and not requested.empty and 'game_key' in frame.columns and 'game_key' in requested.columns:
                keys = set(requested['game_key'].astype(str))
                frame = frame[frame['game_key'].astype(str).isin(keys)].copy().sort_values(['Date','game_key']).reset_index(drop=True)
            if len(frame) == len(requested) and len(frame) >= 1000 and all(c in frame.columns for c in LEGACY_ML_COLUMNS):
                X = frame[LEGACY_ML_COLUMNS].apply(pd.to_numeric, errors='coerce')
                valid = X.notna().all(axis=1)
                frame = frame.loc[valid].reset_index(drop=True)
                X = X.loc[valid].to_numpy(float)
                if len(frame) == len(requested):
                    self.training_source = 'duckdb_parquet_feature_store_subset_safe'
                    return (
                        X, frame['target_home_win'].to_numpy(int),
                        frame['target_total_runs'].to_numpy(float),
                        frame['target_run_diff'].to_numpy(float),
                        pd.to_datetime(frame['Date']).to_numpy(),
                    )
        except Exception as exc:
            print(f'Big Data fallback a CSV: {exc}')

        X, yw, yr, yd, dates = [], [], [], [], []
        hist, hh = {}, {}
        safe = games.copy()
        safe['_day'] = pd.to_datetime(safe['Date'], errors='coerce').dt.normalize()
        for day, daily in safe.groupby('_day', sort=True):
            pending = []
            for _, r in daily.iterrows():
                loc, vis = r.Home, r.Away
                year = int(r.Season)
                hs, away_score = float(r.Home_Score), float(r.Away_Score)
                sy = year - 1
                ol = float(bd.get((loc,sy), self.bat_scale))
                ov = float(bd.get((vis,sy), self.bat_scale))
                pl = float(pdict.get((loc,sy), self.pit_scale))
                pv = float(pdict.get((vis,sy), self.pit_scale))
                X.append(self._feature_row(hist,hh,loc,vis,ol,ov,pl,pv))
                yw.append(int(hs > away_score)); yr.append(hs + away_score); yd.append(hs - away_score); dates.append(day)
                pending.append((loc,vis,hs,away_score))
            for loc,vis,hs,away_score in pending:
                append_game(hist,hh,loc,vis,hs,away_score)
        self.training_source = 'csv_chronological_daily_safe'
        return np.asarray(X,float), np.asarray(yw), np.asarray(yr), np.asarray(yd), np.asarray(dates,dtype='datetime64[ns]')

    def _advanced_matrix(self, df_batting, df_pitching, games):
        if not self.training_source.startswith('duckdb_'):
            return None
        try:
            from .bigdata_mlb import MLBDataWarehouse
            wh = MLBDataWarehouse()
            frame = wh.training_frame().copy()
            requested = wh._normalize_games(games)
            keys = set(requested['game_key'].astype(str))
            frame = frame[frame['game_key'].astype(str).isin(keys)].copy().sort_values(['Date','game_key']).reset_index(drop=True)
            if len(frame) != len(requested):
                return None
            advanced = build_advanced_signal_frame(frame, df_batting, df_pitching)
            return advanced.to_numpy(float) if len(advanced) == len(frame) else None
        except Exception as exc:
            print(f'Advanced signals disabled: {exc}')
            return None

    def _evaluate(self, X, yw, yr, yd, cut):
        runs, diffs = {}, {}
        for family in ('gbr','histgb'):
            rm = self._new_regressor(family); rm.fit(X[:cut], yr[:cut]); rp = rm.predict(X[cut:])
            dm = self._new_regressor(family); dm.fit(X[:cut], yd[:cut]); dp = dm.predict(X[cut:])
            runs[family] = (float(np.mean(np.abs(rp - yr[cut:]))), rp)
            diffs[family] = (float(np.mean(np.abs(dp - yd[cut:]))), dp)
        runs_family = min(runs, key=lambda k: runs[k][0])
        diff_family = min(diffs, key=lambda k: diffs[k][0])
        best = None
        for family in ('logistic','histgb'):
            cm = self._new_classifier(family); cm.fit(X[:cut], yw[:cut]); raw = cm.predict_proba(X[cut:])[:,1]
            for alpha in np.linspace(0.35, 1.0, 66):
                cal = 0.5 + alpha * (raw - 0.5)
                brier = float(np.mean((cal - yw[cut:]) ** 2))
                if best is None or brier < best[0]:
                    best = (brier, family, float(alpha))
        return {
            'brier':best[0], 'classifier':best[1], 'alpha':best[2],
            'runs_family':runs_family, 'diff_family':diff_family,
            'runs_mae':runs[runs_family][0], 'diff_mae':diffs[diff_family][0],
            'runs_pred':runs[runs_family][1], 'diff_pred':diffs[diff_family][1],
        }

    @staticmethod
    def _advanced_wins(base, advanced):
        if advanced is None:
            return False, None
        ratios = [
            advanced['brier']/max(base['brier'],1e-9),
            advanced['runs_mae']/max(base['runs_mae'],1e-9),
            advanced['diff_mae']/max(base['diff_mae'],1e-9),
        ]
        composite = float(np.mean(ratios))
        wins = ((composite <= 0.997 and max(ratios) <= 1.015) or
                (ratios[0] <= 0.995 and ratios[1] <= 1.01 and ratios[2] <= 1.01))
        return bool(wins), round((1.0 - composite) * 100.0, 3)

    @staticmethod
    def _last_dates(games):
        out = {}
        g = games.copy(); g['Date'] = pd.to_datetime(g['Date'], errors='coerce')
        for _, r in g.dropna(subset=['Date']).iterrows():
            d = pd.Timestamp(r.Date).normalize()
            for team in (normalize_team(r.Home), normalize_team(r.Away)):
                if team and (team not in out or d > out[team]):
                    out[team] = d
        return out

    def _live_rest(self, team, game_date=None):
        last = self.last_game_dates.get(normalize_team(team))
        if last is None:
            return 3.0
        return float(np.clip((_as_of_date(game_date) - pd.Timestamp(last).normalize()).days - 1, 0, 7))

    def entrenar(self, df_batting, df_pitching, df_games):
        try:
            if df_batting.empty or df_pitching.empty or df_games.empty:
                return False
            key = _cache_key(df_batting, df_pitching, df_games)
            with _MODEL_CACHE_LOCK:
                cached = _MODEL_CACHE.get(key)
                if cached is not None:
                    _MODEL_CACHE.move_to_end(key)
                    self._restore_cached(cached)
                    return True

            bat_col, pit_col = batting_metric(df_batting), pitching_metric(df_pitching)
            if not bat_col or not pit_col:
                return False
            games = prepare_games(df_games)
            bd, pdict = self._stats_dict(df_batting,bat_col), self._stats_dict(df_pitching,pit_col)
            bv = pd.to_numeric(df_batting[bat_col],errors='coerce').dropna()
            pv = pd.to_numeric(df_pitching[pit_col],errors='coerce').dropna()
            self.bat_scale = float(bv.median()) if len(bv) else 100.0
            self.pit_scale = float(pv.median()) if len(pv) else 4.10
            X,yw,yr,yd,dates = self._training_arrays(df_batting,df_pitching,games,bd,pdict)
            self.training_rows = int(len(X))
            if len(X) < 1000:
                return False
            cut = _date_safe_cut(dates)
            train_dates = pd.to_datetime(pd.Series(dates[:cut])).dt.normalize()
            val_dates = pd.to_datetime(pd.Series(dates[cut:])).dt.normalize()
            if set(train_dates.dropna().unique()).intersection(set(val_dates.dropna().unique())):
                raise RuntimeError('Corte temporal inválido')
            self.validation_cut_date = None if val_dates.empty else str(val_dates.iloc[0].date())

            base_eval = self._evaluate(X,yw,yr,yd,cut)
            advanced_extra = self._advanced_matrix(df_batting,df_pitching,games)
            advanced_eval, X_advanced = None, None
            self.signal_coverage = coverage_report(df_batting,df_pitching)
            eligible = sum(v >= 0.65 for k,v in self.signal_coverage.items() if not k.startswith('context:')) >= 6
            if eligible and advanced_extra is not None and len(advanced_extra) == len(X):
                X_advanced = np.hstack([X,advanced_extra])
                advanced_eval = self._evaluate(X_advanced,yw,yr,yd,cut)
            use_advanced, self.signal_gain = self._advanced_wins(base_eval,advanced_eval)
            chosen = advanced_eval if use_advanced else base_eval
            Xfit = X_advanced if use_advanced else X
            self.signal_set = 'advanced_prior_season' if use_advanced else 'baseline20'
            self.signal_batting = df_batting.copy()
            self.signal_pitching = df_pitching.copy()
            seasons = pd.to_numeric(df_batting.get('Season',pd.Series(dtype=float)),errors='coerce').dropna()
            self.signal_season = int(seasons.max()) if len(seasons) else None
            self.last_game_dates = self._last_dates(games)
            self.history_as_of = max(self.last_game_dates.values()).strftime('%Y-%m-%d') if self.last_game_dates else None

            self.classifier_family = chosen['classifier']; self.runs_family = chosen['runs_family']; self.diff_family = chosen['diff_family']
            self.prob_shrink = chosen['alpha']; self.validation_brier = chosen['brier']
            self.validation_runs_mae = chosen['runs_mae']; self.validation_diff_mae = chosen['diff_mae']
            self.sigma_runs = _calibrate_sigma(chosen['runs_pred'],yr[cut:],'total')
            self.sigma_diff = _calibrate_sigma(chosen['diff_pred'],yd[cut:],'spread')
            self.modelo_ganador = self._new_classifier(self.classifier_family)
            self.modelo_carreras = self._new_regressor(self.runs_family)
            self.modelo_handicap = self._new_regressor(self.diff_family)
            self.modelo_ganador.fit(Xfit,yw); self.modelo_carreras.fit(Xfit,yr); self.modelo_handicap.fit(Xfit,yd)

            hist,hh = {},{}
            for _,r in games.iterrows():
                append_game(hist,hh,r.Home,r.Away,float(r.Home_Score),float(r.Away_Score))
            self.current_history,self.current_h2h = hist,hh
            self.entrenado = True; self.loaded_from_cache = False

            fields = (
                'modelo_ganador','modelo_carreras','modelo_handicap','classifier_family','runs_family','diff_family',
                'bat_scale','pit_scale','sigma_runs','sigma_diff','prob_shrink','validation_brier','validation_runs_mae',
                'validation_diff_mae','validation_cut_date','current_history','current_h2h','training_source','training_rows',
                'signal_set','signal_gain','signal_coverage','signal_batting','signal_pitching','signal_season','last_game_dates','history_as_of'
            )
            cache = {f: deepcopy(getattr(self,f)) if f in _MUTABLE_CACHE_FIELDS else getattr(self,f) for f in fields}
            with _MODEL_CACHE_LOCK:
                _MODEL_CACHE[key] = cache
                _MODEL_CACHE.move_to_end(key)
                while len(_MODEL_CACHE) > _MODEL_CACHE_MAX:
                    _MODEL_CACHE.popitem(last=False)
            return True
        except Exception as exc:
            print(f'Error entrenando ML MLB: {exc}')
            self.entrenado = False
            return False

    def actualizar_resultado(self, loc_abbr, vis_abbr, home_score, away_score, game_date=None):
        # Copy-on-write: cached training state must remain immutable after a live update.
        self.current_history = deepcopy(self.current_history)
        self.current_h2h = deepcopy(self.current_h2h)
        self.last_game_dates = deepcopy(self.last_game_dates)
        loc,vis = normalize_team(loc_abbr),normalize_team(vis_abbr)
        append_game(self.current_history,self.current_h2h,loc,vis,float(home_score),float(away_score))
        if game_date is not None:
            d = _as_of_date(game_date)
            self.last_game_dates[loc] = d; self.last_game_dates[vis] = d
            self.history_as_of = max(self.last_game_dates.values()).strftime('%Y-%m-%d') if self.last_game_dates else None

    @staticmethod
    def _cdf(z):
        return _normal_cdf(z)

    def predecir_partido(self, loc_abbr, vis_abbr, wrc_loc, wrc_vis, xfip_loc, xfip_vis, pf=None, game_date=None):
        try:
            if not self.entrenado:
                raise RuntimeError('Modelo no entrenado')
            as_of = _as_of_date(game_date)
            pred_season = int(as_of.year)
            loc,vis = normalize_team(loc_abbr),normalize_team(vis_abbr)
            base = np.asarray(self._feature_row(
                self.current_history,self.current_h2h,loc,vis,float(wrc_loc),float(wrc_vis),float(xfip_loc),float(xfip_vis)
            ),float)
            rest_l,rest_v = self._live_rest(loc,as_of),self._live_rest(vis,as_of)
            features = base
            if self.signal_set == 'advanced_prior_season':
                features = np.concatenate([
                    base,
                    build_live_signal_row(loc,vis,pred_season,self.signal_batting,self.signal_pitching,rest_l,rest_v),
                ])
            f = features.reshape(1,-1)
            raw = float(self.modelo_ganador.predict_proba(f)[0,1])
            p = self._shrink_probability(raw,self.prob_shrink)
            runs = float(self.modelo_carreras.predict(f)[0])
            diff = float(self.modelo_handicap.predict(f)[0])
            return {
                'Probabilidad_Local':round(p*100,2),'Probabilidad_Visita':round((1-p)*100,2),
                'Probabilidad_Local_Raw':round(raw*100,2),'Prob_Shrink':round(self.prob_shrink,3),
                'Proyeccion_Carreras':round(runs,2),'Proyeccion_Handicap_Local':round(diff,2),
                'Sigma_Carreras':round(self.sigma_runs,3),'Sigma_Handicap':round(self.sigma_diff,3),
                'Modelo_Desde_Cache':bool(self.loaded_from_cache),'Classifier_Family':self.classifier_family,
                'Runs_Family':self.runs_family,'Diff_Family':self.diff_family,'Training_Source':self.training_source,
                'Training_Rows':self.training_rows,'Validation_Cut_Date':self.validation_cut_date,
                'Signal_Set':self.signal_set,'Signal_Gain_Pct':self.signal_gain,'Signal_Coverage':self.signal_coverage,
                'Signal_Season_Used':pred_season-1,'Rest_Days_Local':rest_l,'Rest_Days_Visita':rest_v,
                'Prediction_Date':str(as_of.date()),'History_As_Of':self.history_as_of,
                'Validation_Brier':None if self.validation_brier is None else round(self.validation_brier,5),
                'Validation_Runs_MAE':None if self.validation_runs_mae is None else round(self.validation_runs_mae,4),
                'Validation_Diff_MAE':None if self.validation_diff_mae is None else round(self.validation_diff_mae,4),
            }
        except Exception as exc:
            print(f'Error en predicción ML: {exc}')
            return {
                'Probabilidad_Local':50.0,'Probabilidad_Visita':50.0,'Probabilidad_Local_Raw':50.0,
                'Prob_Shrink':1.0,'Proyeccion_Carreras':8.5,'Proyeccion_Handicap_Local':0.0,
                'Sigma_Carreras':3.5,'Sigma_Handicap':4.2,'Modelo_Desde_Cache':False,
                'Classifier_Family':'fallback','Runs_Family':'fallback','Diff_Family':'fallback',
                'Training_Source':'fallback','Training_Rows':0,'Signal_Set':'fallback',
            }
