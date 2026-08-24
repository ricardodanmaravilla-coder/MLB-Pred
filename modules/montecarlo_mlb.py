import numpy as np
import pandas as pd

from .team_utils import normalize_team


def calcular_factor_clima(viento_mph, direccion_viento, temp_f):
    """Ajuste moderado del entorno de carreras.

    Solo usa la dirección cuando ya viene expresada como Infield/Outfield por la UI.
    """
    try:
        wind = float(viento_mph or 0.0)
        temp = float(temp_f or 72.0)
    except (TypeError, ValueError):
        wind, temp = 0.0, 72.0
    direction = str(direccion_viento or "").lower()
    mult = 1.0
    if "outfield" in direction or "hacia afuera" in direction:
        mult += min(wind, 25.0) * 0.004
    elif "infield" in direction or "hacia adentro" in direction:
        mult -= min(wind, 25.0) * 0.004
    mult += np.clip(temp - 72.0, -30.0, 30.0) * 0.0015
    return float(np.clip(mult, 0.88, 1.12))


def _normalized_games(df_games):
    if df_games is None or df_games.empty:
        return pd.DataFrame()
    g = df_games.copy()
    if "Home" not in g.columns or "Away" not in g.columns:
        return pd.DataFrame()
    g["HomeKey"] = g["Home"].map(normalize_team)
    g["AwayKey"] = g["Away"].map(normalize_team)
    if "Date" in g.columns:
        g["Date"] = pd.to_datetime(g["Date"], errors="coerce")
        g = g.sort_values("Date")
    return g


def obtener_h2h(df_games, loc_abbr, vis_abbr):
    """Se conserva para diagnóstico; ya no participa en la probabilidad final."""
    g = _normalized_games(df_games)
    if g.empty:
        return 50.0
    loc, vis = normalize_team(loc_abbr), normalize_team(vis_abbr)
    rows = g[((g.HomeKey == loc) & (g.AwayKey == vis)) | ((g.HomeKey == vis) & (g.AwayKey == loc))].tail(20)
    if rows.empty:
        return 50.0
    wins = 0
    valid = 0
    for _, r in rows.iterrows():
        try:
            hs, as_ = float(r["Home_Score"]), float(r["Away_Score"])
        except (KeyError, TypeError, ValueError):
            continue
        valid += 1
        if (r.HomeKey == loc and hs > as_) or (r.AwayKey == loc and as_ > hs):
            wins += 1
    return 50.0 if valid == 0 else wins / valid * 100.0


def obtener_carreras_recientes(df_games, equipo_abbr, n=10):
    g = _normalized_games(df_games)
    if g.empty:
        return None
    team = normalize_team(equipo_abbr)
    rows = g[(g.HomeKey == team) | (g.AwayKey == team)].tail(int(n))
    if rows.empty:
        return None
    runs = []
    for _, r in rows.iterrows():
        try:
            if r.HomeKey == team:
                runs.append(float(r["Home_Score"]))
            else:
                runs.append(float(r["Away_Score"]))
        except (KeyError, TypeError, ValueError):
            continue
    return float(np.mean(runs)) if runs else None


def _spread_probability(diff, side, line):
    line = float(line)
    if side == "local":
        return float(np.mean((diff + line) > 0) * 100.0)
    return float(np.mean((-diff + line) > 0) * 100.0)


def simular_partido_mlb(
    local, visita, pitcher_loc_xfip, pitcher_vis_xfip, wrc_loc, wrc_vis,
    bullpen_loc_era, bullpen_vis_era, park_factor, altitud_ft,
    viento_mph, direccion_viento, temp_f, linea_carreras_casino,
    df_games=None, num_simulaciones=50000
):
    """Monte Carlo compatible con la UI histórica, sin fallbacks inventados.

    La firma y las claves principales de salida se mantienen para no romper app_mlb.py.
    """
    if linea_carreras_casino is None or float(linea_carreras_casino) <= 0:
        raise ValueError("Línea de carreras de casino requerida y no disponible.")

    metricas = [wrc_loc, wrc_vis, pitcher_loc_xfip, pitcher_vis_xfip, bullpen_loc_era, bullpen_vis_era, park_factor]
    if any(m is None or pd.isna(m) for m in metricas):
        raise ValueError(f"Datos incompletos para simular {visita} @ {local}.")

    loc_key, vis_key = normalize_team(local), normalize_team(visita)
    recent_loc = obtener_carreras_recientes(df_games, loc_key)
    recent_vis = obtener_carreras_recientes(df_games, vis_key)

    # Se mantienen las escalas históricas de la app, pero se eliminan errores de identidad y parque.
    bat_loc = np.clip(float(wrc_loc) / 100.0, 0.75, 1.25)
    bat_vis = np.clip(float(wrc_vis) / 100.0, 0.75, 1.25)
    sp_loc = np.clip(float(pitcher_loc_xfip) / 4.10, 0.70, 1.30)
    sp_vis = np.clip(float(pitcher_vis_xfip) / 4.10, 0.70, 1.30)
    bp_loc = np.clip(float(bullpen_loc_era) / 4.10, 0.75, 1.30)
    bp_vis = np.clip(float(bullpen_vis_era) / 4.10, 0.75, 1.30)

    climate = calcular_factor_clima(viento_mph, direccion_viento, temp_f)
    park = np.clip(float(park_factor) / 100.0, 0.88, 1.12)

    # Un estadio ofensivo afecta a ambos equipos; no se invierte para la visita.
    pitching_vis = sp_vis * 0.60 + bp_vis * 0.40
    pitching_loc = sp_loc * 0.60 + bp_loc * 0.40
    base_loc = 4.55 * bat_loc * pitching_vis * park * climate
    base_vis = 4.45 * bat_vis * pitching_loc * park * climate

    # Tendencia reciente solo se usa cuando existe; no se sustituye silenciosamente por 4.6.
    exp_loc = base_loc if recent_loc is None else 0.70 * base_loc + 0.30 * recent_loc
    exp_vis = base_vis if recent_vis is None else 0.70 * base_vis + 0.30 * recent_vis
    exp_loc = float(np.clip(exp_loc, 1.5, 8.5))
    exp_vis = float(np.clip(exp_vis, 1.5, 8.5))

    sims = int(np.clip(num_simulaciones, 5000, 100000))
    dispersion = 14.5
    p_loc = dispersion / (dispersion + exp_loc)
    p_vis = dispersion / (dispersion + exp_vis)
    rng = np.random.default_rng()
    c_loc = rng.negative_binomial(dispersion, p_loc, sims)
    c_vis = rng.negative_binomial(dispersion, p_vis, sims)

    # MLB no termina empatado: para empates simulados se usa una ventaja local mínima, no un H2H histórico.
    ties = c_loc == c_vis
    n_ties = int(np.sum(ties))
    if n_ties:
        local_wins = rng.random(n_ties) < 0.53
        c_loc[ties] += local_wins.astype(int)
        c_vis[ties] += (~local_wins).astype(int)

    prob_loc = float(np.mean(c_loc > c_vis) * 100.0)
    prob_vis = 100.0 - prob_loc
    totals = c_loc + c_vis
    diff = c_loc - c_vis
    line = float(linea_carreras_casino)

    runs = {
        "Promedio_Total": round(float(np.mean(totals)), 2),
        f"Over {linea_carreras_casino}": round(float(np.mean(totals > line) * 100.0), 2),
        f"Under {linea_carreras_casino}": round(float(np.mean(totals < line) * 100.0), 2),
        f"Push {linea_carreras_casino}": round(float(np.mean(totals == line) * 100.0), 2),
    }

    # Claves legacy y dinámicas para que la UI pueda consultar la línea real sin fabricar probabilidades.
    for spread in (-3.5, -2.5, -1.5, 1.5, 2.5, 3.5):
        runs[f"Spread Local {spread:+.1f}"] = round(_spread_probability(diff, "local", spread), 2)
        runs[f"Spread Visita {spread:+.1f}"] = round(_spread_probability(diff, "visita", spread), 2)
    runs["Spread Local -1.5"] = runs["Spread Local -1.5"]
    runs["Spread Local +1.5"] = runs["Spread Local +1.5"]
    runs["Spread Visita -1.5"] = runs["Spread Visita -1.5"]
    runs["Spread Visita +1.5"] = runs["Spread Visita +1.5"]

    pyth_exp = (exp_loc + exp_vis) ** 0.285
    pyth_loc = (exp_loc ** pyth_exp) / ((exp_loc ** pyth_exp) + (exp_vis ** pyth_exp)) * 100.0
    h2h = obtener_h2h(df_games, loc_key, vis_key)

    return {
        "Moneyline": {"Gana Local": round(prob_loc, 2), "Gana Visita": round(prob_vis, 2)},
        "Carreras": runs,
        "Metadatos": {
            "Pythagenpat_Loc": round(float(pyth_loc), 2),
            "H2H_Loc": round(float(h2h), 2),
            "H2H_Usado_En_Probabilidad": False,
            "Carreras_Exp_Local": round(exp_loc, 2),
            "Carreras_Exp_Visita": round(exp_vis, 2),
            "Simulaciones": sims,
        },
    }
