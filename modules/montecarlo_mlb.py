import numpy as np
import pandas as pd

def calcular_factor_clima(viento_mph, direccion_viento, temp_f):
    mult_carreras = 1.0
    if "out" in direccion_viento.lower(): mult_carreras += (viento_mph * 0.006)
    elif "in" in direccion_viento.lower(): mult_carreras -= (viento_mph * 0.006)
    diff_temp = temp_f - 72
    mult_carreras += (diff_temp * 0.002)
    return np.clip(mult_carreras, 0.80, 1.20)

def obtener_h2h(df_games, loc_abbr, vis_abbr):
    """Calcula el porcentaje de victorias históricas H2H del equipo local sobre la visita"""
    if df_games is None or df_games.empty:
        return 50.0
    
    try:
        cols = [c.lower() for c in df_games.columns]
        if 'home' in cols and 'away' in cols and ('home_score' in cols or 'homescore' in cols):
            h_col = df_games.columns[cols.index('home')]
            a_col = df_games.columns[cols.index('away')]
            hs_col = df_games.columns[cols.index('home_score') if 'home_score' in cols else cols.index('homescore')]
            as_col = df_games.columns[cols.index('away_score') if 'away_score' in cols else cols.index('awayscore')]
            
            juegos_h2h = df_games[((df_games[h_col] == loc_abbr) & (df_games[a_col] == vis_abbr)) | 
                                  ((df_games[h_col] == vis_abbr) & (df_games[a_col] == loc_abbr))]
            
            if len(juegos_h2h) == 0: return 50.0
            
            victorias_loc = 0
            for _, row in juegos_h2h.iterrows():
                if row[h_col] == loc_abbr and row[hs_col] > row[as_col]: victorias_loc += 1
                elif row[a_col] == loc_abbr and row[as_col] > row[hs_col]: victorias_loc += 1
                
            return (victorias_loc / len(juegos_h2h)) * 100.0
    except:
        pass
    return 50.0

def simular_partido_mlb(
    local, visita, pitcher_loc_xfip, pitcher_vis_xfip, wrc_loc, wrc_vis,
    bullpen_loc_era, bullpen_vis_era, park_factor, altitud_ft,
    viento_mph, direccion_viento, temp_f, linea_carreras_casino,
    df_games=None, num_simulaciones=1000000
):
    # Validación estricta sin números inventados
    if linea_carreras_casino is None or linea_carreras_casino <= 0:
        raise ValueError("Línea de carreras de casino requerida y no disponible.")

    metricas = [wrc_loc, wrc_vis, pitcher_loc_xfip, pitcher_vis_xfip, bullpen_loc_era, bullpen_vis_era, park_factor]
    if any(pd.isna(m) for m in metricas) or None in metricas:
         raise ValueError(f"Datos sabermétricos incompletos para simular {visita} @ {local}.")

    # 1. NORMALIZACIÓN ESTADÍSTICA
    w_loc_f = np.clip(float(wrc_loc) / 100.0, 0.75, 1.25)
    w_vis_f = np.clip(float(wrc_vis) / 100.0, 0.75, 1.25)
    
    p_loc_f = np.clip(float(pitcher_loc_xfip) / 4.10, 0.70, 1.30)
    p_vis_f = np.clip(float(pitcher_vis_xfip) / 4.10, 0.70, 1.30)
    
    bp_loc_f = np.clip(float(bullpen_loc_era) / 4.10, 0.75, 1.30)
    bp_vis_f = np.clip(float(bullpen_vis_era) / 4.10, 0.75, 1.30)

    factor_clima = calcular_factor_clima(viento_mph, direccion_viento, temp_f)
    pf_loc = np.clip(float(park_factor) / 100.0, 0.85, 1.15)

    # 2. PROYECCIÓN DE CARRERAS (BaseRuns adaptado)
    carreras_exp_loc = 4.3 * w_loc_f * ((p_vis_f * 0.65) + (bp_vis_f * 0.35)) * pf_loc * factor_clima
    carreras_exp_vis = 4.3 * w_vis_f * ((p_loc_f * 0.65) + (bp_loc_f * 0.35)) * (1.0 / pf_loc) * factor_clima

    if np.isnan(carreras_exp_loc) or carreras_exp_loc <= 0: carreras_exp_loc = 4.3
    if np.isnan(carreras_exp_vis) or carreras_exp_vis <= 0: carreras_exp_vis = 4.3

    # 3. MODELO 1: PYTHAGENPAT
    exponente = (carreras_exp_loc + carreras_exp_vis) ** 0.285
    prob_pyth_loc = (carreras_exp_loc ** exponente) / ((carreras_exp_loc ** exponente) + (carreras_exp_vis ** exponente)) * 100

    # 4. MODELO 2: H2H HISTÓRICO
    loc_abbr = local if len(local) <= 3 else local[:3].upper()
    vis_abbr = visita if len(visita) <= 3 else visita[:3].upper()
    prob_h2h_loc = obtener_h2h(df_games, loc_abbr, vis_abbr)

    # 5. MODELO 3: MONTECARLO POISSON
    c_loc_sim = np.random.poisson(carreras_exp_loc, num_simulaciones)
    c_vis_sim = np.random.poisson(carreras_exp_vis, num_simulaciones)
    
    empates = (c_loc_sim == c_vis_sim)
    desempate = np.random.rand(np.sum(empates)) > 0.47 
    c_loc_sim[empates] += desempate.astype(int)
    c_vis_sim[empates] += (~desempate).astype(int)

    prob_mc_loc = np.mean(c_loc_sim > c_vis_sim) * 100
    
    # Promediar los modelos y limitar
    prob_final_loc = (prob_pyth_loc + prob_h2h_loc + prob_mc_loc) / 3.0
    prob_final_loc = np.clip(prob_final_loc, 35.0, 65.0)
    prob_final_vis = 100.0 - prob_final_loc

    totales = c_loc_sim + c_vis_sim
    dif_carreras = c_loc_sim - c_vis_sim

    # Probabilidades reales de hándicap calculadas por simulación
    prob_spread_loc_minus_1_5 = np.mean(dif_carreras >= 2) * 100
    prob_spread_vis_plus_1_5 = np.mean(dif_carreras >= -1) * 100

    return {
        "Moneyline": {
            "Gana Local": round(prob_final_loc, 2),
            "Gana Visita": round(prob_final_vis, 2)
        },
        "Carreras": {
            "Promedio_Total": round(np.mean(totales), 2),
            f"Over {linea_carreras_casino}": round(np.mean(totales > linea_carreras_casino) * 100, 2),
            f"Under {linea_carreras_casino}": round(np.mean(totales < linea_carreras_casino) * 100, 2),
            "Spread -1.5 Local": round(prob_spread_loc_minus_1_5, 2),
            "Spread +1.5 Visita": round(prob_spread_vis_plus_1_5, 2),
        },
        "Metadatos": {
            "Pythagenpat_Loc": round(prob_pyth_loc, 2),
            "H2H_Loc": round(prob_h2h_loc, 2)
        }
    }
