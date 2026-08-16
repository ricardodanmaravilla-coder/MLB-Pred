import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor

class PredictorMLMLB:
    def __init__(self):
        # Usamos Random Forest para predecir al ganador y Gradient Boosting para las carreras
        self.modelo_ganador = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
        self.modelo_carreras = GradientBoostingRegressor(n_estimators=150, max_depth=5, random_state=42)
        # Modelo dedicado para el Hándicap (Diferencial de Carreras)
        self.modelo_handicap = GradientBoostingRegressor(n_estimators=150, max_depth=5, random_state=42)
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
            y_diff = []

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

                # NORMALIZACIÓN ESTRICTA: Las variables de entrenamiento deben ser idénticas
                # a las variables que pasamos en predecir_partido.
                diff_wrc = float(wrc_l - wrc_v) / 20.0
                diff_xfip = float(xfip_v - xfip_l) / 2.0
                diff_racha = float(racha_l - racha_v) / 5.0

                feat = [diff_wrc, diff_xfip, diff_racha, float(racha_l)]
                
                X.append(feat)
                y_win.append(1 if g_loc > g_vis else 0)
                y_runs.append(g_loc + g_vis)
                y_diff.append(g_loc - g_vis)

            # Entrenar solo si tenemos una muestra robusta
            if len(X) > 100:
                self.modelo_ganador.fit(X, y_win)
                self.modelo_carreras.fit(X, y_runs)
                self.modelo_handicap.fit(X, y_diff)
                self.entrenado = True
                return True
                
        except Exception as e:
            print(f"Error entrenando ML MLB: {e}")
            
        return False

    def predecir_partido(self, loc_abbr, vis_abbr, wrc_loc, wrc_vis, xfip_loc, xfip_vis, pf=None):
        try:
            racha_loc = 2.5
            racha_vis = 2.5
            
            if not self.df_games.empty:
                juegos_loc = self.df_games[(self.df_games['Home'] == loc_abbr) | (self.df_games['Away'] == loc_abbr)].tail(5)
                if not juegos_loc.empty:
                    victorias_loc = sum((juegos_loc['Home'] == loc_abbr) & (juegos_loc['Home_Score'] > juegos_loc['Away_Score'])) + \
                                    sum((juegos_loc['Away'] == loc_abbr) & (juegos_loc['Away_Score'] > juegos_loc['Home_Score']))
                    racha_loc = victorias_loc
                
                juegos_vis = self.df_games[(self.df_games['Home'] == vis_abbr) | (self.df_games['Away'] == vis_abbr)].tail(5)
                if not juegos_vis.empty:
                    victorias_vis = sum((juegos_vis['Home'] == vis_abbr) & (juegos_vis['Home_Score'] > juegos_vis['Away_Score'])) + \
                                    sum((juegos_vis['Away'] == vis_abbr) & (juegos_vis['Away_Score'] > juegos_vis['Home_Score']))
                    racha_vis = victorias_vis

            # Normalización estricta de variables para evitar desbordamiento en los árboles de decisión
            diff_wrc = float(wrc_loc - wrc_vis) / 20.0
            diff_xfip = float(xfip_vis - xfip_loc) / 2.0
            diff_racha = float(racha_loc - racha_vis) / 5.0

            features = [[diff_wrc, diff_xfip, diff_racha, float(racha_loc)]]
            
            # Obtención de probabilidades base
            probs = self.modelo_ganador.predict_proba(features)[0]
            
            # Suavización matemática (Evita ceros absolutos y 100% forzados)
            p_local_suavizada = (probs[1] * 0.85) + 0.075 
            p_visita_suavizada = (probs[0] * 0.85) + 0.075

            carreras_pred = self.modelo_carreras.predict(features)[0] 
            handicap_pred = self.modelo_handicap.predict(features)[0] 
            
            return {
                'Probabilidad_Local': round(p_local_suavizada * 100, 2),
                'Probabilidad_Visita': round(p_visita_suavizada * 100, 2),
                'Proyeccion_Carreras': round(float(carreras_pred), 2), 
                'Proyeccion_Handicap_Local': round(float(handicap_pred), 2) 
            }
        except Exception as e:
            print(f"Error en predicción ML: {e}")
            return {'Probabilidad_Local': 50.0, 'Probabilidad_Visita': 50.0, 'Proyeccion_Carreras': 8.5, 'Proyeccion_Handicap_Local': 0.0}
            
