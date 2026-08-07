import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor

class PredictorMLMLB:
    def __init__(self):
        # Usamos Random Forest para predecir al ganador y Gradient Boosting para las carreras
        self.modelo_ganador = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
        self.modelo_carreras = GradientBoostingRegressor(n_estimators=150, max_depth=5, random_state=42)
        self.entrenado = False
        self.df_games = pd.DataFrame()

    def entrenar(self, df_batting, df_pitching, df_games):
        """
        Entrena el modelo usando RESULTADOS REALES, cruzando el score con
        la Sabermetría avanzada y calculando tendencias de rachas (Momentum).
        """
        try:
            if df_batting.empty or df_pitching.empty or df_games.empty:
                return False
                
            self.df_games = df_games.copy()
            self.df_games['Date'] = pd.to_datetime(self.df_games['Date'], errors='coerce')
            self.df_games = self.df_games.sort_values('Date')

            X = []
            y_win = []
            y_runs = []

            # Diccionarios rápidos para buscar sabermetría (O(1) lookup)
            bat_dict = df_batting.set_index('Team')['wRC+'].to_dict()
            pit_dict = df_pitching.set_index('Team')['xFIP'].to_dict()

            # Diccionario para rastrear la forma reciente (últimos 5 resultados)
            equipos = pd.concat([self.df_games['Home'], self.df_games['Away']]).unique()
            forma_reciente = {equipo: [] for equipo in equipos}

            for idx, row in self.df_games.iterrows():
                loc = row.get('Home')
                vis = row.get('Away')
                g_loc = row.get('Home_Score')
                g_vis = row.get('Away_Score')
                
                if pd.isna(g_loc) or pd.isna(g_vis) or loc not in bat_dict or vis not in bat_dict: 
                    continue
                
                wrc_l = bat_dict.get(loc, 100.0)
                wrc_v = bat_dict.get(vis, 100.0)
                xfip_l = pit_dict.get(loc, 4.0)
                xfip_v = pit_dict.get(vis, 4.0)

                # Calcular la racha actual antes de jugar este partido (Momentum)
                racha_l = sum(forma_reciente.get(loc, [-1])[-5:]) if len(forma_reciente.get(loc, [])) > 0 else 2.5
                racha_v = sum(forma_reciente.get(vis, [-1])[-5:]) if len(forma_reciente.get(vis, [])) > 0 else 2.5
                
                # Actualizar el historial de resultados para el próximo ciclo
                if g_loc > g_vis:
                    forma_reciente[loc].append(1)
                    forma_reciente[vis].append(0)
                else:
                    forma_reciente[loc].append(0)
                    forma_reciente[vis].append(1)

                # FEATURES (Variables de entrenamiento):
                # 1. Ventaja Ofensiva | 2. Ventaja de Pitcheo | 3. Momentum Local | 4. Momentum Visita
                feat = [wrc_l - wrc_v, xfip_v - xfip_l, racha_l, racha_v]
                
                X.append(feat)
                y_win.append(1 if g_loc > g_vis else 0)
                y_runs.append(g_loc + g_vis)

            # Entrenar solo si tenemos una muestra robusta
            if len(X) > 100:
                self.modelo_ganador.fit(X, y_win)
                self.modelo_carreras.fit(X, y_runs)
                self.entrenado = True
                return True
                
        except Exception as e:
            print(f"Error entrenando ML MLB: {e}")
            
        return False

    def predecir_partido(self, local, visita, wrc_loc, wrc_vis, xfip_loc, xfip_vis, park_factor=100):
        if not self.entrenado:
            return {"Prob_Local_ML": 50.0, "Prob_Visita_ML": 50.0, "Carreras_Proyectadas_ML": 8.5}

        # Calcular la racha en vivo del equipo de cara al partido de HOY
        racha_l = 2.5
        racha_v = 2.5
        
        if not self.df_games.empty:
            # Racha Local
            juegos_loc = self.df_games[(self.df_games['Home'] == local) | (self.df_games['Away'] == local)].tail(5)
            wins_l = sum([1 for _, row in juegos_loc.iterrows() if (row['Home'] == local and row['Home_Score'] > row['Away_Score']) or (row['Away'] == local and row['Away_Score'] > row['Home_Score'])])
            racha_l = wins_l

            # Racha Visita
            juegos_vis = self.df_games[(self.df_games['Home'] == visita) | (self.df_games['Away'] == visita)].tail(5)
            wins_v = sum([1 for _, row in juegos_vis.iterrows() if (row['Home'] == visita and row['Home_Score'] > row['Away_Score']) or (row['Away'] == visita and row['Away_Score'] > row['Home_Score'])])
            racha_v = wins_v

        # Alimentar al modelo entrenado
        feat = [[wrc_loc - wrc_vis, xfip_vis - xfip_loc, racha_l, racha_v]]
        
        probs = self.modelo_ganador.predict_proba(feat)[0]
        prob_loc = probs[1] * 100 if len(probs) > 1 else 50.0
        
        runs_est = float(self.modelo_carreras.predict(feat)[0]) * (park_factor / 100.0)

        return {
            "Prob_Local_ML": round(prob_loc, 2),
            "Prob_Visita_ML": round(100 - prob_loc, 2),
            "Carreras_Proyectadas_ML": round(runs_est, 2)
        }
