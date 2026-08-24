import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from .team_utils import normalize_team


class PredictorMLMLB:
    """Modelo prepartido MLB con calibración cronológica de moneyline."""

    FEATURES = [
        "ops_home", "ops_away", "era_home", "era_away",
        "win5_home", "win5_away", "rf5_home", "rf5_away",
        "ra5_home", "ra5_away", "rf10_home", "rf10_away",
        "ra10_home", "ra10_away", "home_field",
    ]

    def __init__(self):
        self.modelo_ganador = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=8,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1,
        )
        self.calibrador_ganador = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
        self.modelo_carreras = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.035, loss="huber", random_state=42,
        )
        self.modelo_handicap = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.035, loss="huber", random_state=42,
        )
        self.entrenado = False
        self.calibrado = False
        self.df_games = pd.DataFrame()
        self.residuales_carreras = np.array([])
        self.residuales_handicap = np.array([])

    @staticmethod
    def _clean_games(df_games):
        g = df_games.copy()
        g["Date"] = pd.to_datetime(g["Date"], errors="coerce")
        g["Home"] = g["Home"].map(normalize_team)
        g["Away"] = g["Away"].map(normalize_team)
        g["Home_Score"] = pd.to_numeric(g["Home_Score"], errors="coerce")
        g["Away_Score"] = pd.to_numeric(g["Away_Score"], errors="coerce")
        if "Season" not in g.columns:
            g["Season"] = g["Date"].dt.year
        g["Season"] = pd.to_numeric(g["Season"], errors="coerce")
        return g.dropna(subset=["Date", "Home", "Away", "Home_Score", "Away_Score", "Season"]).sort_values("Date").reset_index(drop=True)

    @staticmethod
    def _stats_dicts(df_batting, df_pitching):
        bat = df_batting.copy(); pit = df_pitching.copy()
        bat["Team"] = bat["Team"].map(normalize_team); pit["Team"] = pit["Team"].map(normalize_team)
        bat["Season"] = pd.to_numeric(bat["Season"], errors="coerce")
        pit["Season"] = pd.to_numeric(pit["Season"], errors="coerce")
        bat["ops"] = pd.to_numeric(bat.get("ops"), errors="coerce")
        pit["ERA"] = pd.to_numeric(pit.get("ERA", pit.get("era")), errors="coerce")
        ops = bat.dropna(subset=["Team", "Season", "ops"]).set_index(["Team", "Season"])["ops"].to_dict()
        era = pit.dropna(subset=["Team", "Season", "ERA"]).set_index(["Team", "Season"])["ERA"].to_dict()
        return ops, era

    @staticmethod
    def _rolling_state(history, team, n):
        rows = history.get(team, [])[-n:]
        if not rows:
            return 0.5, 4.5, 4.5
        return float(np.mean([r[0] for r in rows])), float(np.mean([r[1] for r in rows])), float(np.mean([r[2] for r in rows]))

    def preparar_dataset(self, df_batting, df_pitching, df_games):
        g = self._clean_games(df_games)
        ops_dict, era_dict = self._stats_dicts(df_batting, df_pitching)
        history, rows = {}, []
        for _, r in g.iterrows():
            h, a, season = r.Home, r.Away, int(r.Season)
            # Estadística de temporada previa: evita usar el agregado final de la
            # misma temporada para predecir un partido que ocurrió antes.
            ops_h = float(ops_dict.get((h, season - 1), 0.720)); ops_a = float(ops_dict.get((a, season - 1), 0.720))
            era_h = float(era_dict.get((h, season - 1), 4.20)); era_a = float(era_dict.get((a, season - 1), 4.20))
            wh5, rfh5, rah5 = self._rolling_state(history, h, 5); wa5, rfa5, raa5 = self._rolling_state(history, a, 5)
            _, rfh10, rah10 = self._rolling_state(history, h, 10); _, rfa10, raa10 = self._rolling_state(history, a, 10)
            rows.append({
                "Date": r.Date, "Home": h, "Away": a,
                "ops_home": ops_h, "ops_away": ops_a, "era_home": era_h, "era_away": era_a,
                "win5_home": wh5, "win5_away": wa5, "rf5_home": rfh5, "rf5_away": rfa5,
                "ra5_home": rah5, "ra5_away": raa5, "rf10_home": rfh10, "rf10_away": rfa10,
                "ra10_home": rah10, "ra10_away": raa10, "home_field": 1.0,
                "Target_Win": int(r.Home_Score > r.Away_Score),
                "Target_Runs": float(r.Home_Score + r.Away_Score), "Target_Diff": float(r.Home_Score - r.Away_Score),
            })
            history.setdefault(h, []).append((int(r.Home_Score > r.Away_Score), float(r.Home_Score), float(r.Away_Score)))
            history.setdefault(a, []).append((int(r.Away_Score > r.Home_Score), float(r.Away_Score), float(r.Home_Score)))
        return pd.DataFrame(rows)

    def entrenar_preparado(self, prepared):
        if prepared is None or len(prepared) < 1000:
            return False
        n = len(prepared)
        cut = max(800, int(n * 0.80)) if n >= 1500 else n
        base = prepared.iloc[:cut]
        calib = prepared.iloc[cut:]
        Xb = base[self.FEATURES].astype(float)
        self.modelo_ganador.fit(Xb, base["Target_Win"].astype(int))
        self.calibrado = False
        if len(calib) >= 150 and calib["Target_Win"].nunique() == 2:
            Xc = calib[self.FEATURES].astype(float)
            raw = self.modelo_ganador.predict_proba(Xc)[:, 1].reshape(-1, 1)
            self.calibrador_ganador.fit(raw, calib["Target_Win"].astype(int))
            self.calibrado = True
        # Para regresión todo prepared es pasado respecto al bloque de test.
        X = prepared[self.FEATURES].astype(float)
        self.modelo_carreras.fit(X, prepared["Target_Runs"].astype(float))
        self.modelo_handicap.fit(X, prepared["Target_Diff"].astype(float))
        self.entrenado = True
        return True

    def predict_proba_features(self, X):
        raw = self.modelo_ganador.predict_proba(X[self.FEATURES].astype(float))[:, 1]
        if self.calibrado:
            return self.calibrador_ganador.predict_proba(raw.reshape(-1, 1))[:, 1]
        return raw

    def entrenar(self, df_batting, df_pitching, df_games):
        try:
            if df_batting.empty or df_pitching.empty or df_games.empty:
                return False
            self.df_games = self._clean_games(df_games)
            prep = self.preparar_dataset(df_batting, df_pitching, df_games)
            if len(prep) < 1200:
                return False
            # Residuales estrictamente OOS para convertir regresiones a P(O/U).
            cut = max(1000, int(len(prep) * 0.80))
            train, calib = prep.iloc[:cut], prep.iloc[cut:]
            temp_runs = GradientBoostingRegressor(n_estimators=160, max_depth=3, learning_rate=0.04, loss="huber", random_state=43)
            temp_diff = GradientBoostingRegressor(n_estimators=160, max_depth=3, learning_rate=0.04, loss="huber", random_state=44)
            Xt = train[self.FEATURES].astype(float); Xc = calib[self.FEATURES].astype(float)
            temp_runs.fit(Xt, train["Target_Runs"].astype(float)); temp_diff.fit(Xt, train["Target_Diff"].astype(float))
            self.residuales_carreras = calib["Target_Runs"].to_numpy(float) - temp_runs.predict(Xc)
            self.residuales_handicap = calib["Target_Diff"].to_numpy(float) - temp_diff.predict(Xc)
            return self.entrenar_preparado(prep)
        except Exception as exc:
            print(f"Error entrenando ML MLB V2: {exc}")
            self.entrenado = False
            return False

    def _current_roll(self, team, n):
        g = self.df_games
        rows = g[(g.Home == team) | (g.Away == team)].tail(n)
        if rows.empty:
            return 0.5, 4.5, 4.5
        wins, rf, ra = [], [], []
        for _, r in rows.iterrows():
            if r.Home == team:
                wins.append(int(r.Home_Score > r.Away_Score)); rf.append(r.Home_Score); ra.append(r.Away_Score)
            else:
                wins.append(int(r.Away_Score > r.Home_Score)); rf.append(r.Away_Score); ra.append(r.Home_Score)
        return float(np.mean(wins)), float(np.mean(rf)), float(np.mean(ra))

    def construir_features_actuales(self, loc_abbr, vis_abbr, ops_loc, ops_vis, era_loc, era_vis):
        h, a = normalize_team(loc_abbr), normalize_team(vis_abbr)
        if not h or not a:
            raise ValueError("Equipo MLB no reconocido")
        wh5, rfh5, rah5 = self._current_roll(h, 5); wa5, rfa5, raa5 = self._current_roll(a, 5)
        _, rfh10, rah10 = self._current_roll(h, 10); _, rfa10, raa10 = self._current_roll(a, 10)
        return pd.DataFrame([{
            "ops_home": float(ops_loc), "ops_away": float(ops_vis), "era_home": float(era_loc), "era_away": float(era_vis),
            "win5_home": wh5, "win5_away": wa5, "rf5_home": rfh5, "rf5_away": rfa5,
            "ra5_home": rah5, "ra5_away": raa5, "rf10_home": rfh10, "rf10_away": rfa10,
            "ra10_home": rah10, "ra10_away": raa10, "home_field": 1.0,
        }])[self.FEATURES]

    def predecir_partido(self, loc_abbr, vis_abbr, ops_loc, ops_vis, era_loc, era_vis, pf=None):
        if not self.entrenado:
            raise RuntimeError("ML MLB no entrenado")
        X = self.construir_features_actuales(loc_abbr, vis_abbr, ops_loc, ops_vis, era_loc, era_vis)
        p_home = float(self.predict_proba_features(X)[0])
        runs = float(self.modelo_carreras.predict(X)[0]); diff = float(self.modelo_handicap.predict(X)[0])
        return {
            "Probabilidad_Local": round(p_home * 100.0, 2), "Probabilidad_Visita": round((1.0 - p_home) * 100.0, 2),
            "Proyeccion_Carreras": round(runs, 2), "Proyeccion_Handicap_Local": round(diff, 2),
            "Calibrada": bool(self.calibrado),
        }

    def prob_total(self, proyeccion, linea, side="over"):
        if len(self.residuales_carreras) < 100:
            return None
        actual = float(proyeccion) + self.residuales_carreras
        if side == "over": return float(np.mean(actual > float(linea)) * 100.0)
        if side == "under": return float(np.mean(actual < float(linea)) * 100.0)
        return None

    def prob_runline(self, proyeccion_diff, linea_local):
        if len(self.residuales_handicap) < 100:
            return None
        actual = float(proyeccion_diff) + self.residuales_handicap
        return float(np.mean((actual + float(linea_local)) > 0.0) * 100.0)
