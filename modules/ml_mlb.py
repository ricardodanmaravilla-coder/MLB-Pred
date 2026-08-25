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


# Streamlit reruns the script after practically every widget interaction.  The old
# implementation recreated and retrained all three sklearn estimators on every rerun.
# Keep a tiny in-process cache keyed by stable dataset signatures so subsequent
# interactions reuse the already fitted estimators instead of retraining them.
_MODEL_CACHE = OrderedDict()
_MODEL_CACHE_LOCK = threading.RLock()
_MODEL_CACHE_MAX = 2


def _frame_signature(df, important_columns):
    """Content-sensitive signature so changed middle rows cannot reuse stale estimators."""
    if df is None or df.empty:
        return (0, 0, "")
    cols = [c for c in important_columns if c in df.columns]
    if not cols:
        cols = list(df.columns)
    hashed = pd.util.hash_pandas_object(df[cols], index=True).values.tobytes()
    digest = hashlib.sha256(hashed).hexdigest()
    return (int(len(df)), int(len(df.columns)), digest)


def _cache_key(df_batting, df_pitching, df_games):
    return (
        _frame_signature(df_batting, ('Team', 'Season', 'OPS_Index', 'wRC+')),
        _frame_signature(df_pitching, ('Team', 'Season', 'ERA', 'xFIP')),
        _frame_signature(df_games, ('Date', 'Season', 'Home', 'Away', 'Home_Score', 'Away_Score')),
    )


def _normal_cdf(z):
    return 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))


def _calibrate_sigma(pred, actual, market='spread'):
    """Calibrate residual scale on chronological holdout market probabilities.

    A plain residual standard deviation can materially overstate confidence on MLB
    run lines, especially +1.5.  Select a modest scale multiplier on the validation
    block by Brier score.  No future observations are used.
    """
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if len(pred) < 100:
        return float(max(1.0, np.std(actual - pred)))

    base = float(max(1.0, np.std(actual - pred)))
    best_sigma, best_score = base, float('inf')

    if market == 'spread':
        # Score both sides of the canonical MLB ±1.5 market so calibration is not
        # biased toward whichever side has the naturally higher base rate.
        outcomes = [
            (actual + 1.5 > 0).astype(float),
            (-actual + 1.5 > 0).astype(float),
        ]
        projections = [pred + 1.5, -pred + 1.5]
    else:
        # Totals commonly live around 7.0-10.0; using several anchors makes sigma
        # useful when today's line differs from exactly 8.5.
        outcomes, projections = [], []
        for line in (7.5, 8.5, 9.5):
            outcomes.append((actual > line).astype(float))
            projections.append(pred - line)
            outcomes.append((actual < line).astype(float))
            projections.append(line - pred)

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


class PredictorMLMLB:
    """Pre-game MLB model: real historical results + prior-season team strength."""

    def __init__(self):
        self.modelo_ganador = self._new_classifier()
        self.modelo_carreras = GradientBoostingRegressor(
            n_estimators=140, max_depth=2, learning_rate=0.035,
            loss='huber', random_state=42
        )
        self.modelo_handicap = GradientBoostingRegressor(
            n_estimators=140, max_depth=2, learning_rate=0.035,
            loss='huber', random_state=42
        )
        self.entrenado = False
        self.bat_scale, self.pit_scale = 100.0, 4.10
        self.current_history, self.current_h2h = {}, {}
        self.sigma_runs, self.sigma_diff = 3.5, 4.2
        self.prob_shrink = 1.0
        self.loaded_from_cache = False

    @staticmethod
    def _new_classifier():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.35, max_iter=2000, solver='lbfgs', random_state=42)
        )

    @staticmethod
    def _stats_dict(df, col):
        x = df.copy()
        x['Team'] = x['Team'].map(normalize_team)
        x['Season'] = pd.to_numeric(x['Season'], errors='coerce')
        x[col] = pd.to_numeric(x[col], errors='coerce')
        return x.dropna(subset=['Team', 'Season', col]).set_index(['Team', 'Season'])[col].to_dict()

    def _feature_row(self, hist, h2h, loc, vis, off_l, off_v, pit_l, pit_v):
        w5l, rf5l, ra5l, rd5l = team_state(hist, loc, 5)
        w5v, rf5v, ra5v, rd5v = team_state(hist, vis, 5)
        w20l, rf20l, ra20l, rd20l = team_state(hist, loc, 20)
        w20v, rf20v, ra20v, rd20v = team_state(hist, vis, 20)
        hwin, hrd, hn = h2h_state(h2h, loc, vis, 12)
        return [
            w5l, w5v, w20l, w20v,
            rf5l, rf5v, ra5l, ra5v,
            rd5l, rd5v, rd20l, rd20v,
            hwin, hrd, min(hn, 12) / 12.0,
            float(off_l) / max(self.bat_scale, 1e-6),
            float(off_v) / max(self.bat_scale, 1e-6),
            float(pit_l) / max(self.pit_scale, 1e-6),
            float(pit_v) / max(self.pit_scale, 1e-6),
            1.0,
        ]

    @staticmethod
    def _shrink_probability(raw_prob, alpha):
        return float(np.clip(0.5 + float(alpha) * (float(raw_prob) - 0.5), 0.01, 0.99))

    def _restore_cached(self, cached):
        self.modelo_ganador = cached['modelo_ganador']
        self.modelo_carreras = cached['modelo_carreras']
        self.modelo_handicap = cached['modelo_handicap']
        self.bat_scale = cached['bat_scale']
        self.pit_scale = cached['pit_scale']
        self.sigma_runs = cached['sigma_runs']
        self.sigma_diff = cached['sigma_diff']
        self.prob_shrink = cached['prob_shrink']
        # Rolling state is mutable; each Predictor instance receives an independent
        # copy while heavy sklearn estimators remain shared/read-only.
        self.current_history = deepcopy(cached['current_history'])
        self.current_h2h = deepcopy(cached['current_h2h'])
        self.entrenado = True
        self.loaded_from_cache = True

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

            bat_col = 'OPS_Index' if 'OPS_Index' in df_batting.columns else 'wRC+'
            pit_col = 'ERA' if 'ERA' in df_pitching.columns else 'xFIP'
            if bat_col not in df_batting or pit_col not in df_pitching:
                return False

            games = prepare_games(df_games)
            bd = self._stats_dict(df_batting, bat_col)
            pdict = self._stats_dict(df_pitching, pit_col)
            bv = pd.to_numeric(df_batting[bat_col], errors='coerce').dropna()
            pv = pd.to_numeric(df_pitching[pit_col], errors='coerce').dropna()
            self.bat_scale = float(bv.median()) if len(bv) else 100.0
            self.pit_scale = float(pv.median()) if len(pv) else 4.10

            X, yw, yr, yd = [], [], [], []
            hist, hh = {}, {}
            for _, r in games.iterrows():
                loc, vis = r.Home, r.Away
                year = int(r.Season)
                hs, as_ = float(r.Home_Score), float(r.Away_Score)
                sy = year - 1
                ol = float(bd.get((loc, sy), self.bat_scale))
                ov = float(bd.get((vis, sy), self.bat_scale))
                pl = float(pdict.get((loc, sy), self.pit_scale))
                pv_ = float(pdict.get((vis, sy), self.pit_scale))
                X.append(self._feature_row(hist, hh, loc, vis, ol, ov, pl, pv_))
                yw.append(int(hs > as_))
                yr.append(hs + as_)
                yd.append(hs - as_)
                append_game(hist, hh, loc, vis, hs, as_)

            if len(X) < 1000:
                return False

            X = np.asarray(X, float)
            yw = np.asarray(yw)
            yr = np.asarray(yr)
            yd = np.asarray(yd)
            cut = max(100, int(len(X) * 0.80))
            if cut >= len(X) - 50:
                cut = max(100, len(X) - 50)

            # Chronological validation: fit only on the earlier block, calibrate on
            # later unseen games, then refit on all available historical games.
            self.modelo_carreras.fit(X[:cut], yr[:cut])
            self.modelo_handicap.fit(X[:cut], yd[:cut])
            val_runs_pred = self.modelo_carreras.predict(X[cut:])
            val_diff_pred = self.modelo_handicap.predict(X[cut:])
            self.sigma_runs = _calibrate_sigma(val_runs_pred, yr[cut:], market='total')
            self.sigma_diff = _calibrate_sigma(val_diff_pred, yd[cut:], market='spread')

            temp_classifier = self._new_classifier()
            temp_classifier.fit(X[:cut], yw[:cut])
            raw_cal = temp_classifier.predict_proba(X[cut:])[:, 1]
            best_alpha, best_brier = 1.0, float('inf')
            for alpha in np.linspace(0.35, 1.00, 66):
                cal = 0.5 + alpha * (raw_cal - 0.5)
                score = float(np.mean((cal - yw[cut:]) ** 2))
                if score < best_brier:
                    best_brier, best_alpha = score, float(alpha)
            self.prob_shrink = best_alpha

            self.modelo_ganador = self._new_classifier()
            self.modelo_ganador.fit(X, yw)
            self.modelo_carreras.fit(X, yr)
            self.modelo_handicap.fit(X, yd)
            self.current_history, self.current_h2h = hist, hh
            self.entrenado = True
            self.loaded_from_cache = False

            cache_value = {
                'modelo_ganador': self.modelo_ganador,
                'modelo_carreras': self.modelo_carreras,
                'modelo_handicap': self.modelo_handicap,
                'bat_scale': self.bat_scale,
                'pit_scale': self.pit_scale,
                'sigma_runs': self.sigma_runs,
                'sigma_diff': self.sigma_diff,
                'prob_shrink': self.prob_shrink,
                'current_history': deepcopy(hist),
                'current_h2h': deepcopy(hh),
            }
            with _MODEL_CACHE_LOCK:
                _MODEL_CACHE[key] = cache_value
                _MODEL_CACHE.move_to_end(key)
                while len(_MODEL_CACHE) > _MODEL_CACHE_MAX:
                    _MODEL_CACHE.popitem(last=False)
            return True

        except Exception as e:
            print(f'Error entrenando ML MLB: {e}')
            self.entrenado = False
            return False

    def actualizar_resultado(self, loc_abbr, vis_abbr, home_score, away_score):
        """Advance rolling/H2H state after a completed game without retraining estimators."""
        loc, vis = normalize_team(loc_abbr), normalize_team(vis_abbr)
        append_game(
            self.current_history, self.current_h2h,
            loc, vis, float(home_score), float(away_score)
        )

    @staticmethod
    def _cdf(z):
        return _normal_cdf(z)

    def predecir_partido(self, loc_abbr, vis_abbr, wrc_loc, wrc_vis, xfip_loc, xfip_vis, pf=None):
        try:
            if not self.entrenado:
                raise RuntimeError('Modelo no entrenado')
            loc, vis = normalize_team(loc_abbr), normalize_team(vis_abbr)
            f = np.asarray([
                self._feature_row(
                    self.current_history, self.current_h2h,
                    loc, vis, float(wrc_loc), float(wrc_vis),
                    float(xfip_loc), float(xfip_vis)
                )
            ], float)
            raw_local = float(self.modelo_ganador.predict_proba(f)[0, 1])
            p_local = self._shrink_probability(raw_local, self.prob_shrink)
            p_vis = 1.0 - p_local
            runs = float(self.modelo_carreras.predict(f)[0])
            diff = float(self.modelo_handicap.predict(f)[0])
            return {
                'Probabilidad_Local': round(p_local * 100, 2),
                'Probabilidad_Visita': round(p_vis * 100, 2),
                'Probabilidad_Local_Raw': round(raw_local * 100, 2),
                'Prob_Shrink': round(self.prob_shrink, 3),
                'Proyeccion_Carreras': round(runs, 2),
                'Proyeccion_Handicap_Local': round(diff, 2),
                'Sigma_Carreras': round(self.sigma_runs, 3),
                'Sigma_Handicap': round(self.sigma_diff, 3),
                'Modelo_Desde_Cache': bool(self.loaded_from_cache),
            }
        except Exception as e:
            print(f'Error en predicción ML: {e}')
            return {
                'Probabilidad_Local': 50.0,
                'Probabilidad_Visita': 50.0,
                'Probabilidad_Local_Raw': 50.0,
                'Prob_Shrink': 1.0,
                'Proyeccion_Carreras': 8.5,
                'Proyeccion_Handicap_Local': 0.0,
                'Sigma_Carreras': 3.5,
                'Sigma_Handicap': 4.2,
                'Modelo_Desde_Cache': False,
            }
