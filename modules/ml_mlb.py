import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor

from .team_utils import normalize_team


class PredictorMLMLB:
    """Predictor compatible con la UI estable, con entrenamiento prepartido coherente."""

    def __init__(self):
        self.modelo_ganador = RandomForestClassifier(
            n_estimators=150, max_depth=7, min_samples_leaf=8,
            class_weight="balanced_subsample", random_state=42, n_jobs=1,
        )
        self.modelo_carreras = GradientBoostingRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.04, loss="huber", random_state=42
        )
        self.modelo_handicap = GradientBoostingRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.04, loss="huber", random_state=42
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
    def _form(history, team, n=5):
        rows = history.get(team, [])[-n:]
        return 0.5 if not rows else float(np.mean(rows))

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
                form_l = self._form(history, loc, 5)
                form_v = self._form(history, vis, 5)

                off_l_n = off_l / max(self.bat_scale, 1e-6)
                off_v_n = off_v / max(self.bat_scale, 1e-6)
                pit_l_n = pit_l / max(self.pit_scale, 1e-6)
                pit_v_n = pit_v / max(self.pit_scale, 1e-6)
                matchup_l = off_l_n / max(pit_v_n, 0.25)
                matchup_v = off_v_n / max(pit_l_n, 0.25)
                X.append([off_l_n, off_v_n, pit_l_n, pit_v_n, form_l, form_v, matchup_l, matchup_v])
                y_win.append(int(g_loc > g_vis))
                y_runs.append(g_loc + g_vis)
                y_diff.append(g_loc - g_vis)
                history.setdefault(loc, []).append(int(g_loc > g_vis))
                history.setdefault(vis, []).append(int(g_vis > g_loc))

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

    def _recent_form(self, team):
        if self.df_games.empty:
            return 0.5
        key = normalize_team(team)
        rows = self.df_games[(self.df_games["Home"] == key) | (self.df_games["Away"] == key)].tail(5)
        if rows.empty:
            return 0.5
        wins = 0
        for _, r in rows.iterrows():
            if (r["Home"] == key and r["Home_Score"] > r["Away_Score"]) or (r["Away"] == key and r["Away_Score"] > r["Home_Score"]):
                wins += 1
        return wins / len(rows)

    def predecir_partido(self, loc_abbr, vis_abbr, wrc_loc, wrc_vis, xfip_loc, xfip_vis, pf=None):
        try:
            if not self.entrenado:
                raise RuntimeError("Modelo no entrenado")
            form_l = self._recent_form(loc_abbr)
            form_v = self._recent_form(vis_abbr)
            off_l_n = float(wrc_loc) / max(self.bat_scale, 1e-6)
            off_v_n = float(wrc_vis) / max(self.bat_scale, 1e-6)
            pit_l_n = float(xfip_loc) / max(self.pit_scale, 1e-6)
            pit_v_n = float(xfip_vis) / max(self.pit_scale, 1e-6)
            matchup_l = off_l_n / max(pit_v_n, 0.25)
            matchup_v = off_v_n / max(pit_l_n, 0.25)
            features = np.asarray([[off_l_n, off_v_n, pit_l_n, pit_v_n, form_l, form_v, matchup_l, matchup_v]], dtype=float)
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
