import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor

class PredictorMLMLB:
    def __init__(self):
        self.modelo_ganador = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
        self.modelo_carreras = GradientBoostingRegressor(n_estimators=150, max_depth=5, random_state=42)
        self.modelo_handicap = GradientBoostingRegressor(n_estimators=150, max_depth=5, random_state=42)
        self.entrenado = False
        self.df_games = pd.DataFrame()

    def entrenar(self, df_batting, df_pitching, df_games):
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

            bat_dict = df_batting.set_index('Team')['wRC+'].to_dict()
            pit_dict = df_pitching.set_index('Team')['xFIP'].to_dict()

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

                racha_l = sum(forma_reciente.get(loc, [-1])[-5:]) if len(forma_reciente.get(loc, [])) > 0 else 2.5
                racha_v = sum(forma_reciente.get(vis, [-1])[-5:]) if len(forma_reciente.get(vis, [])) > 0 else 2.5
                
                if g_loc > g_vis:
                    forma_reciente[loc].append(1)
                    forma_reciente[vis].append(0)
                else:
                    forma_reciente[loc].append(0)
                    forma_reciente[vis].append(1)

                # CORRECCIÓN: Usar valores absolutos normalizados en lugar de restas
                wrc_l_norm = float(wrc_l) / 100.0
                wrc_v_norm = float(wrc_v) / 100.0
                xfip_l_norm = float(xfip_l) / 4.0
                xfip_v_norm = float(xfip_v) / 4.0
                r_l_norm = float(racha_l) / 5.0
                r_v_norm = float(racha_v) / 5.0

                feat = [wrc_l_norm, wrc_v_norm, xfip_l_norm, xfip_v_norm, r_l_norm, r_v_norm]
                
                X.append(feat)
                y_win.append(1 if g_loc > g_vis else 0)
                y_runs.append(g_loc + g_vis)
                y_diff.append(g_loc - g_vis)

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

            # CORRECCIÓN: Entradas absolutas idénticas a la función de entrenamiento
            wrc_l_norm = float(wrc_loc) / 100.0
            wrc_v_norm = float(wrc_vis) / 100.0
            xfip_l_norm = float(xfip_loc) / 4.0
            xfip_v_norm = float(xfip_vis) / 4.0
            r_l_norm = float(racha_loc) / 5.0
            r_v_norm = float(racha_vis) / 5.0

            features = [[wrc_l_norm, wrc_v_norm, xfip_l_norm, xfip_v_norm, r_l_norm, r_v_norm]]
            
            probs = self.modelo_ganador.predict_proba(features)[0]
            
            # Suavización más ligera para permitir alertas fuertes de Moneyline
            p_local_suavizada = (probs[1] * 0.94) + 0.03 
            p_visita_suavizada = (probs[0] * 0.94) + 0.03

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
            
