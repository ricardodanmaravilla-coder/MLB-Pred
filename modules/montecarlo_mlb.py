import numpy as np
import pandas as pd

def calcular_factor_clima(viento_mph, direccion_viento, temp_f):
    mult_carreras = 1.0
    if "out" in direccion_viento.lower():
        mult_carreras += (viento_mph * 0.005)
    elif "in" in direccion_viento.lower():
        mult_carreras -= (viento_mph * 0.005)
    
    diff_temp = temp_f - 72
    mult_carreras += (diff_temp * 0.002)
    return max(0.85, min(1.15, mult_carreras))

def simular_partido_mlb(
    local, visita,
    pitcher_loc_xfip, pitcher_vis_xfip,
    wrc_loc, wrc_vis,
    bullpen_loc_era, bullpen_vis_era,
    park_factor, altitud_ft,
    viento_mph, direccion_viento, temp_f,
    linea_carreras_casino,
    num_simulaciones=100000
):
    if linea_carreras_casino is None or linea_carreras_casino <= 0:
        linea_carreras_casino = 8.5 # Fallback seguro

    # 1. REGRESIÓN EXTREMA A LA MEDIA (Shrinkage Sabermétrico)
    # Suavizamos las métricas para que el impacto por entrada sea marginal y realista
    peso_b = 0.15 # La ofensiva solo impacta un 15% por encima de la media
    peso_p = 0.15 # El pitcheo solo impacta un 15% por encima de la media

    wrc_loc_f = 1.0 + ((wrc_loc / 100.0) - 1.0) * peso_b
    wrc_vis_f = 1.0 + ((wrc_vis / 100.0) - 1.0) * peso_b

    p_loc_f = 1.0 + ((pitcher_loc_xfip / 4.10) - 1.0) * peso_p
    p_vis_f = 1.0 + ((pitcher_vis_xfip / 4.10) - 1.0) * peso_p

    bp_loc_f = 1.0 + ((bullpen_loc_era / 4.10) - 1.0) * peso_p
    bp_vis_f = 1.0 + ((bullpen_vis_era / 4.10) - 1.0) * peso_p

    # Factores ambientales
    mult_estadio = (park_factor / 100.0)
    mult_clima = calcular_factor_clima(viento_mph, direccion_viento, temp_f)
    
    # 2. EXPECTATIVA BASE
    base_inning = 0.45 * mult_estadio * mult_clima

    # Lambdas "crudas" (Carreras esperadas por entrada)
    raw_loc_start = base_inning * wrc_loc_f * p_vis_f
    raw_vis_start = base_inning * wrc_vis_f * p_loc_f
    raw_loc_bp = base_inning * wrc_loc_f * bp_vis_f
    raw_vis_bp = base_inning * wrc_vis_f * bp_loc_f

    # 3. ANCLAJE A LAS VEGAS (Vegas Anchoring)
    # Calculamos cuántas carreras espera anotar el simulador en total
    expected_loc_total = (raw_loc_start * 6) + (raw_loc_bp * 3)
    expected_vis_total = (raw_vis_start * 6) + (raw_vis_bp * 3)
    expected_total_game = expected_loc_total + expected_vis_total

    # Ajustamos obligatoriamente nuestra matemática para que empate con la línea del Casino
    # De este modo evitamos que Poisson "infle" el marcador irrealmente.
    ajuste_vegas = linea_carreras_casino / expected_total_game if expected_total_game > 0 else 1.0

    # Lambdas finales ajustadas a la realidad del mercado
    L_loc_s = raw_loc_start * ajuste_vegas
    L_vis_s = raw_vis_start * ajuste_vegas
    L_loc_b = raw_loc_bp * ajuste_vegas
    L_vis_b = raw_vis_bp * ajuste_vegas

    # 4. SIMULACIÓN POISSON
    carreras_loc_sim = np.sum(np.random.poisson(L_loc_s, (num_simulaciones, 6)), axis=1) + \
                       np.sum(np.random.poisson(L_loc_b, (num_simulaciones, 3)), axis=1)

    carreras_vis_sim = np.sum(np.random.poisson(L_vis_s, (num_simulaciones, 6)), axis=1) + \
                       np.sum(np.random.poisson(L_vis_b, (num_simulaciones, 3)), axis=1)

    # 5. RESOLUCIÓN DE EMPATES (Extra Innings)
    empates = (carreras_loc_sim == carreras_vis_sim)
    if np.any(empates):
        # El equipo local históricamente gana ~52% de los juegos que van a extra innings
        desempate_local = np.random.rand(np.sum(empates)) > 0.48 
        carreras_loc_sim[empates] += desempate_local.astype(int)
        carreras_vis_sim[empates] += (~desempate_local).astype(int)

    # 6. CÁLCULO DE PROBABILIDADES
    ganador_local = np.mean(carreras_loc_sim > carreras_vis_sim) * 100
    ganador_visita = np.mean(carreras_vis_sim > carreras_loc_sim) * 100
    
    totales = carreras_loc_sim + carreras_vis_sim

    return {
        "Moneyline": {
            "Gana Local": round(ganador_local, 2),
            "Gana Visita": round(ganador_visita, 2)
        },
        "Carreras": {
            "Promedio_Total": round(np.mean(totales), 2),
            f"Over {linea_carreras_casino}": round(np.mean(totales > linea_carreras_casino) * 100, 2),
            f"Under {linea_carreras_casino}": round(np.mean(totales < linea_carreras_casino) * 100, 2),
        }
    }
