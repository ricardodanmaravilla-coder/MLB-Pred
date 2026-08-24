import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .team_utils import normalize_team


class PredictorMLMLB:
    """Modelo MLB compatible con la UI estable usando solo features reproducibles prepartido."""

    def __init__(self):
        self.modelo_ganador = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.35, max_iter=2000, solver="lbfgs", random_state=42),
        )
        self.modelo_carreras = GradientBoostingRegressor(
            n_estimators=140, max_depth=2, learning_rate=0.035, loss="huber", random_state=42
        )
        self.modelo_handicap = GradientBoostingRegressor(
            n_estimators=140, max_depth=2, learning_rate=0.035, loss="huber", random_state=42
        )
        self.entrenado = False
        self.df_games = pd.DataFrame()
        self.bat_scale = 100.0
        self.pit_scale = 4.10

    @staticmethod
    def _normalize_games(df_games):
        g = df_games.copy()
        g["Date"] = pd.to_datetime(g.get("Date"), errors="coerce")
        g["Home"] = g["Home"].map(normalize_team)
        g["Away"] = g["Away"].map(normalize_team)
        g["Home_Score"] = pd.to_numeric(g["Home_Score"], errors="coerce")
        g["Away_Score"] = pd.to_numeric(g["Away_Score"], errors="coerce")
        if "Season" not in g.columns:
            g["Season"] = g["Date"].dt.year
        g["Season"] = pd.to_numeric(g["Season"], errors="coerce")
        return g.dropna(subset=["Date", "Home", "Away", "Home_Score", "Away_Score", "Season"]).sort_values("Date").reset_index(drop=True)

    @staticmethod
    def _stats_dict(df, value_col):
        x = df.copy()
        x["Team"] = x["Team"].map(normalize_team)
        x["Season"] = pd.to_numeric(x["Season"], errors="coerce")
        x[value_col] = pd.to_numeric(x[value_col], errors="coerce")
        x = x.dropna(subset=["Team", "Season", value_col])
        return x.set_index(["Team", "Season"])[value_col].to_dict()

    @staticmethod
    def _state(history, team, n):
        rows = history.get(team, [])[-n:]
        if not rows:
            return 0.5, 4.5, 4.5, 0.0
        wins = np.mean([r[0] for r in rows])
        rf = np.mean([r[1] for r in rows])
        ra = np.mean([r[2] for r in rows])
        return float(wins), float(rf), float(ra), float(rf - ra)

    def _feature_row(self, history, loc, vis, off_l, off_v, pit_l, pit_v):
        w5_l, rf5_l, ra5_l, rd5_l = self._state(history, loc, 5)
        w5_v, rf5_v, ra5_v, rd5_v = self._state(history, vis, 5)
        w20_l, rf20_l, ra20_l, rd20_l = self._state(history, loc, 20)
        w20_v, rf20_v, ra20_v, rd20_v = self._state(history, vis, 20)

        off_l_n = float(off_l) / max(self.bat_scale, 1e-6)
        off_v_n = float(off_v) / max(self.bat_scale, 1e-6)
        pit_l_n = float(pit_l) / max(self.pit_scale, 1e-6)
        pit_v_n = float(pit_v) / max(self.pit_scale, 1e-6)

        return [
            w5_l, w5_v, w20_l, w20_v,
            rf5_l, rf5_v, ra5_l, ra5_v,
            rd5_l, rd5_v, rd20_l, rd20_v,
            off_l_n, off_v_n, pit_l_n, pit_v_n,
            1.0,
        ]

    def entrenar(self, df_batting, df_pitching, df_games):
        try:
            if df_batting.empty or df_pitching.empty or df_games.empty:
                return False
            if "wRC+" not in df_batting.columns or "xFIP" not in df_pitching.columns:
                return False

            self.df_games = self._normalize_games(df_games)
            bat_dict = self._stats_dict(df_batting, "wRC+")
            pit_dict = self._stats_dict(df_pitching, "xFIP")
            bat_vals = pd.to_numeric(df_batting["wRC+"], errors="coerce").dropna()
            pit_vals = pd.to_numeric(df_pitching["xFIP"], errors="coerce").dropna()
            self.bat_scale = float(bat_vals.median()) if not bat_vals.empty else 100.0
            self.pit_scale = float(pit_vals.median()) if not pit_vals.empty else 4.10

            X, y_win, y_runs, y_diff = [], [], [], []
            history = {}
            for _, row in self.df_games.iterrows():
                loc, vis = row["Home"], row["Away"]
                year = int(row["Season"])
                g_loc, g_vis = float(row["Home_Score"]), float(row["Away_Score"])

                stats_year = year - 1
                off_l = float(bat_dict.get((loc, stats_year), self.bat_scale))
                off_v = float(bat_dict.get((vis, stats_year), self.bat_scale))
                pit_l = float(pit_dict.get((loc, stats_year), self.pit_scale))
                pit_v = float(pit_dict.get((vis, stats_year), self.pit_scale))

                X.append(self._feature_row(history, loc, vis, off_l, off_v, pit_l, pit_v))
                y_win.append(int(g_loc > g_vis))
                y_runs.append(g_loc + g_vis)
                y_diff.append(g_loc - g_vis)

                history.setdefault(loc, []).append((int(g_loc > g_vis), g_loc, g_vis))
                history.setdefault(vis, []).append((int(g_vis > g_loc), g_vis, g_loc))

            if len(X) < 1000:
                return False
            X = np.asarray(X, dtype=float)
            self.modelo_ganador.fit(X, np.asarray(y_win, dtype=int))
            self.modelo_carreras.fit(X, np.asarray(y_runs, dtype=float))
            self.modelo_handicap.fit(X, np.asarray(y_diff, dtype=float))
            self.entrenado = True
            return True
        except Exception as e:
            print(f"Error entrenando ML MLB: {e}")
            self.entrenado = False
            return False

    def _history_from_games(self):
        history = {}
        for _, r in self.df_games.iterrows():
            h, a = r["Home"], r["Away"]
            hs, as_ = float(r["Home_Score"]), float(r["Away_Score"])
            history.setdefault(h, []).append((int(hs > as_), hs, as_))
            history.setdefault(a, []).append((int(as_ > hs), as_, hs))
        return history

    def predecir_partido(self, loc_abbr, vis_abbr, wrc_loc, wrc_vis, xfip_loc, xfip_vis, pf=None):
        try:
            if not self.entrenado:
                raise RuntimeError("Modelo no entrenado")
            loc, vis = normalize_team(loc_abbr), normalize_team(vis_abbr)
            history = self._history_from_games()
            features = np.asarray([
                self._feature_row(history, loc, vis, float(wrc_loc), float(wrc_vis), float(xfip_loc), float(xfip_vis))
            ], dtype=float)
            probs = self.modelo_ganador.predict_proba(features)[0]
            return {
                "Probabilidad_Local": round(float(probs[1]) * 100.0, 2),
                "Probabilidad_Visita": round(float(probs[0]) * 100.0, 2),
                "Proyeccion_Carreras": round(float(self.modelo_carreras.predict(features)[0]), 2),
                "Proyeccion_Handicap_Local": round(float(self.modelo_handicap.predict(features)[0]), 2),
            }
        except Exception as e:
            print(f"Error en predicción ML: {e}")
            return {"Probabilidad_Local": 50.0, "Probabilidad_Visita": 50.0, "Proyeccion_Carreras": 8.5, "Proyeccion_Handicap_Local": 0.0}
