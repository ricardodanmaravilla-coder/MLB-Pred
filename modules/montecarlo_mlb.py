import numpy as np
import pandas as pd

def calcular_factor_clima(viento_mph, direccion_viento, temp_f):
    """
    Ajusta la expectativa de carreras y hits según el clima real en vivo.
    """
    mult_carreras = 1.0
    mult_hits = 1.0
    
    if "out" in direccion_viento.lower():
        mult_carreras += (viento_mph * 0.015)
        mult_hits += (viento_mph * 0.008)
    elif "in" in direccion_viento.lower():
        mult_carreras -= (viento_mph * 0.012)
        mult_hits -= (viento_mph * 0.006)
        
    diff_temp = temp_f - 72
    mult_carreras += (diff_temp * 0.003)
    
    return max(0.7, min(1.4, mult_carreras)), max(0.8, min(1.3, mult_hits))

import pandas as pd
import numpy as np

def simular_partido_mlb(
    local, visita,
    pitcher_loc_xfip, pitcher_vis_xfip,
    wrc_loc, wrc_vis,
    bullpen_loc_era, bullpen_vis_era,
    park_factor, altitud_ft,
    viento_mph, direccion_viento, temp_f,
    linea_carreras_casino,
    num_simulaciones=500000
):
    """
    Simulador Montecarlo optimizado con factores lógicos estabilizados para evitar sesgos extremos.
    """
    if linea_carreras_casino is None or linea_carreras_casino <= 0:
        raise ValueError("⚠️ Se requiere una línea de carreras válida del casino.")

    # Factores ambientales y de parque ajustados a escala moderada
    mult_estadio = (park_factor / 100.0) + (altitud_ft / 5280.0 * 0.04)
    
    mult_carreras = 1.0
    if "out" in direccion_viento.lower():
        mult_carreras += (viento_mph * 0.008)
    elif "in" in direccion_viento.lower():
        mult_carreras -= (viento_mph * 0.008)
    
    diff_temp = temp_f - 72
    mult_carreras += (diff_temp * 0.002)
    mult_carreras = max(0.8, min(1.25, mult_carreras))

    base_lambda_inning = 0.48 * mult_estadio * mult_carreras

    # Normalización basada en desviaciones relativas respecto a la media de la liga (wRC+ 100, xFIP/ERA 4.10)
    wrc_loc_factor = wrc_loc / 100.0
    wrc_vis_factor = wrc_vis / 100.0

    pitcher_vis_factor = pitcher_vis_xfip / 4.10
    pitcher_loc_factor = pitcher_loc_xfip / 4.10

    bullpen_vis_factor = bullpen_vis_era / 4.10
    bullpen_loc_factor = bullpen_loc_era / 4.10

    # Cálculo de lambdas por sección del juego (Abridores y Bullpen)
    lambda_loc_starter = max(0.20, base_lambda_inning * wrc_loc_factor * pitcher_vis_factor)
    lambda_vis_starter = max(0.20, base_lambda_inning * wrc_vis_factor * pitcher_loc_factor)

    lambda_loc_bullpen = max(0.20, base_lambda_inning * wrc_loc_factor * bullpen_vis_factor)
    lambda_vis_bullpen = max(0.20, base_lambda_inning * wrc_vis_factor * bullpen_loc_factor)

    # Simulación vectorial masiva mediante distribución de Poisson
    carreras_loc_sim = np.sum(np.random.poisson(lambda_loc_starter, (num_simulaciones, 6)), axis=1) + \
                       np.sum(np.random.poisson(lambda_loc_bullpen, (num_simulaciones, 3)), axis=1)

    carreras_vis_sim = np.sum(np.random.poisson(lambda_vis_starter, (num_simulaciones, 6)), axis=1) + \
                       np.sum(np.random.poisson(lambda_vis_bullpen, (num_simulaciones, 3)), axis=1)

    # Resolución de empates en extra innings con probabilidad justa (50/50)
    empates = (carreras_loc_sim == carreras_vis_sim)
    if np.any(empates):
        desempate = np.random.rand(np.sum(empates)) > 0.50
        carreras_loc_sim[empates] += desempate.astype(int)
        carreras_vis_sim[empates] += (~desempate).astype(int)

    ganador_local = np.mean(carreras_loc_sim > carreras_vis_sim) * 100
    ganador_visita = np.mean(carreras_vis_sim > carreras_loc_sim) * 100
    cover_runline_loc = np.mean((carreras_loc_sim - carreras_vis_sim) > 1.5) * 100

    totales_carreras = carreras_loc_sim + carreras_vis_sim

    return {
        "Moneyline": {
            "Gana Local": round(ganador_local, 2),
            "Gana Visita": round(ganador_visita, 2)
        },
        "Run_Line": {
            "Local -1.5": round(cover_runline_loc, 2),
            "Visita +1.5": round(100 - cover_runline_loc, 2)
        },
        "Carreras": {
            "Promedio_Total": round(np.mean(totales_carreras), 2),
            f"Over {linea_carreras_casino}": round(np.mean(totales_carreras > linea_carreras_casino) * 100, 2),
            f"Under {linea_carreras_casino}": round(np.mean(totales_carreras < linea_carreras_casino) * 100, 2),
        }
    }
