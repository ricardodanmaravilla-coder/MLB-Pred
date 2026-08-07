def simular_partido_mlb(
    local, visita,
    pitcher_loc_xfip, pitcher_vis_xfip,
    wrc_loc, wrc_vis,
    bullpen_loc_era, bullpen_vis_era,
    park_factor=100, altitud_ft=0,
    viento_mph=0, direccion_viento="None", temp_f=72,
    linea_carreras_casino=8.5,  # <--- Agregamos la línea real del casino
    num_simulaciones=10000
):
    # ... (todo el cálculo matemático de Poisson se queda igual) ...
    
    # 4. Cálculo de Probabilidades Dinámicas
    ganador_local = np.mean(carreras_loc_sim > carreras_vis_sim) * 100
    ganador_visita = np.mean(carreras_vis_sim > carreras_loc_sim) * 100
    
    # Run Line (Hándicap estándar de béisbol es -1.5)
    cover_runline_loc = np.mean((carreras_loc_sim - carreras_vis_sim) > 1.5) * 100
    
    # Totales Combinados cruzados contra la línea del casino
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
            f"Over {linea_carreras_casino}": round(np.mean(totales_carreras > linea_carreras_casino) * 100, 2), # <--- Dinámico
            f"Under {linea_carreras_casino}": round(np.mean(totales_carreras < linea_carreras_casino) * 100, 2),
        }
    }
