import numpy as np
import pandas as pd

def calcular_factor_clima(viento_mph, direccion_viento, temp_f):
    """
    Ajusta la expectativa de carreras y hits según el clima real en vivo.
    """
    mult_carreras = 1.0
    mult_hits = 1.0
    
    # Impacto del viento
    if "out" in direccion_viento.lower():
        mult_carreras += (viento_mph * 0.015)  # +1.5% por cada mph hacia afuera
        mult_hits += (viento_mph * 0.008)
    elif "in" in direccion_viento.lower():
        mult_carreras -= (viento_mph * 0.012)
        mult_hits -= (viento_mph * 0.006)
        
    # Impacto de temperatura (Base estándar biológica de 72°F)
    diff_temp = temp_f - 72
    mult_carreras += (diff_temp * 0.003)
    
    return max(0.7, min(1.4, mult_carreras)), max(0.8, min(1.3, mult_hits))

def simular_partido_mlb(
    local, visita,
    pitcher_loc_xfip, pitcher_vis_xfip,
    wrc_loc, wrc_vis,
    bullpen_loc_era, bullpen_vis_era,
    park_factor, altitud_ft,
    viento_mph, direccion_viento, temp_f,
    linea_carreras_casino,
    num_simulaciones=1000000
):
    """
    Simulador Montecarlo entrada por entrada para MLB.
    """
    if linea_carreras_casino is None:
        raise ValueError("⚠️ Error crítico: Se requiere la línea de carreras real del casino.")

    # 1. Factores de ajuste por Estadio y Clima real
    mult_estadio = (park_factor / 100.0) + (altitud_ft / 10000.0 * 0.05)
    f_clima_carreras, f_clima_hits = calcular_factor_clima(viento_mph, direccion_viento, temp_f)
    
    # 2. Expectativa de carreras por entrada (Lambda de Poisson protegida con max(0.1, ...))
    base_lambda = max(0.1, 0.50 * mult_estadio * f_clima_carreras)
    
    # Entradas 1-6 (Abridor) vs Entradas 7-9 (Bullpen)
    lambda_loc_starter = max(0.05, base_lambda * (wrc_loc / 100.0) * (pitcher_vis_xfip / 4.10))
    lambda_vis_starter = max(0.05, base_lambda * (wrc_vis / 100.0) * (pitcher_loc_xfip / 4.10))
    
    lambda_loc_bullpen = max(0.05, base_lambda * (wrc_loc / 100.0) * (bullpen_loc_era / 4.10))
    lambda_vis_bullpen = max(0.05, base_lambda * (wrc_vis / 100.0) * (bullpen_loc_era / 4.10))

    # Media de hits por partido protegida
    lambda_hits_loc = max(0.5, 8.5 * (wrc_loc / 100.0) * (pitcher_vis_xfip / 4.10) * f_clima_hits)
    lambda_hits_vis = max(0.5, 8.5 * (wrc_vis / 100.0) * (pitcher_loc_xfip / 4.10) * f_clima_hits)

    carreras_loc_sim = np.zeros(num_simulaciones)
    carreras_vis_sim = np.zeros(num_simulaciones)
    hits_loc_sim = np.random.poisson(lambda_hits_loc, num_simulaciones)
    hits_vis_sim = np.random.poisson(lambda_hits_vis, num_simulaciones)

    # 3. Bucle de Simulación Montecarlo
    for i in range(num_simulaciones):
        c_loc = np.sum(np.random.poisson(lambda_loc_starter, 6))
        c_vis = np.sum(np.random.poisson(lambda_vis_starter, 6))
        
        c_loc += np.sum(np.random.poisson(lambda_loc_bullpen, 3))
        c_vis += np.sum(np.random.poisson(lambda_vis_bullpen, 3))
        
        if c_loc == c_vis:
            if np.random.rand() > 0.48:  
                c_loc += 1
            else:
                c_vis += 1
                
        carreras_loc_sim[i] = c_loc
        carreras_vis_sim[i] = c_vis

    # 4. Cálculo de Probabilidades Dinámicas
    ganador_local = np.mean(carreras_loc_sim > carreras_vis_sim) * 100
    ganador_visita = np.mean(carreras_vis_sim > carreras_loc_sim) * 100
    
    cover_runline_loc = np.mean((carreras_loc_sim - carreras_vis_sim) > 1.5) * 100
    
    totales_carreras = carreras_loc_sim + carreras_vis_sim
    totales_hits = hits_loc_sim + hits_vis_sim

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
        },
        "Hits": {
            "Promedio_Total": round(np.mean(totales_hits), 2)
        }
    }
