import numpy as np
import pandas as pd

from .team_utils import normalize_team


def calcular_factor_clima(viento_mph, direccion_viento, temp_f):
    """Ajuste acotado. Solo aplica dirección si ya viene expresada como in/out.

    Una dirección cardinal por sí sola no determina si el viento sopla hacia el
    outfield; eso depende de la orientación del estadio.
    """
    mult = 1.0
    try:
        wind = max(0.0, float(viento_mph or 0.0))
        temp = float(temp_f or 72.0)
        direction = str(direccion_viento or "").lower()
        if "out" in direction or "afuera" in direction:
            mult += min(wind, 25.0) * 0.004
        elif "in" in direction or "adentro" in direction:
            mult -= min(wind, 25.0) * 0.004
        mult += np.clip(temp - 72.0, -30.0, 30.0) * 0.0015
    except Exception:
        return 1.0
    return float(np.clip(mult, 0.90, 1.10))


def _clean_games(df_games):
    if df_games is None or df_games.empty:
        return pd.DataFrame()
    g = df_games.copy()
    g["Home"] = g["Home"].map(normalize_team)
    g["Away"] = g["Away"].map(normalize_team)
    g["Date"] = pd.to_datetime(g.get("Date"), errors="coerce")
    g["Home_Score"] = pd.to_numeric(g.get("Home_Score"), errors="coerce")
    g["Away_Score"] = pd.to_numeric(g.get("Away_Score"), errors="coerce")
    return g.dropna(subset=["Home", "Away", "Home_Score", "Away_Score"]).sort_values("Date")


def obtener_tendencia_reciente(df_games, equipo, n=10):
    g = _clean_games(df_games)
    team = normalize_team(equipo)
    if g.empty or not team:
        return {"rf": 4.5, "ra": 4.5, "win_pct": 0.5, "n": 0}
    rows = g[(g.Home == team) | (g.Away == team)].tail(n)
    if rows.empty:
        return {"rf": 4.5, "ra": 4.5, "win_pct": 0.5, "n": 0}
    rf, ra, wins = [], [], []
    for _, r in rows.iterrows():
        if r.Home == team:
            rf.append(float(r.Home_Score)); ra.append(float(r.Away_Score)); wins.append(r.Home_Score > r.Away_Score)
        else:
            rf.append(float(r.Away_Score)); ra.append(float(r.Home_Score)); wins.append(r.Away_Score > r.Home_Score)
    return {"rf": float(np.mean(rf)), "ra": float(np.mean(ra)), "win_pct": float(np.mean(wins)), "n": len(rows)}


def _runline_prob(diff, line, side="home"):
    line = float(line)
    if side == "home":
        wins = (diff + line) > 0.0
        pushes = np.isclose(diff + line, 0.0)
    else:
        wins = ((-diff) + line) > 0.0
        pushes = np.isclose((-diff) + line, 0.0)
    return float(np.mean(wins) * 100.0), float(np.mean(pushes) * 100.0)


def simular_partido_mlb(
    local, visita,
    pitcher_loc_xfip=None, pitcher_vis_xfip=None,
    wrc_loc=None, wrc_vis=None,
    bullpen_loc_era=None, bullpen_vis_era=None,
    park_factor=100.0, altitud_ft=0,
    viento_mph=0, direccion_viento="", temp_f=72,
    linea_carreras_casino=None, df_games=None,
    num_simulaciones=200000,
    ops_loc=None, ops_vis=None,
    pitcher_loc_era=None, pitcher_vis_era=None,
    spread_loc=None, spread_vis=None,
):
    """Simulación V2 basada en métricas que realmente existen en el repositorio.

    Los parámetros legacy wrc/xFIP se aceptan temporalmente para compatibilidad,
    pero se interpretan como OPS escalado y ERA respectivamente. No se publica
    ninguna métrica falsa como wRC+ o xFIP.
    """
    if linea_carreras_casino is None or float(linea_carreras_casino) <= 0:
        raise ValueError("Línea real de carreras requerida")

    h, a = normalize_team(local), normalize_team(visita)
    if not h or not a:
        raise ValueError("Equipo MLB no reconocido")

    # Compatibilidad con datos V1: wRC+ era en realidad OPS*100 y xFIP era ERA.
    if ops_loc is None and wrc_loc is not None:
        ops_loc = float(wrc_loc) / 100.0 if float(wrc_loc) > 2 else float(wrc_loc)
    if ops_vis is None and wrc_vis is not None:
        ops_vis = float(wrc_vis) / 100.0 if float(wrc_vis) > 2 else float(wrc_vis)
    if pitcher_loc_era is None:
        pitcher_loc_era = pitcher_loc_xfip
    if pitcher_vis_era is None:
        pitcher_vis_era = pitcher_vis_xfip

    required = [ops_loc, ops_vis, pitcher_loc_era, pitcher_vis_era, bullpen_loc_era, bullpen_vis_era, park_factor]
    if any(x is None or pd.isna(x) for x in required):
        raise ValueError("Datos ofensivos/pitcheo incompletos: NO BET")

    recent_h = obtener_tendencia_reciente(df_games, h, 10)
    recent_a = obtener_tendencia_reciente(df_games, a, 10)

    league_ops = 0.720
    league_era = 4.20
    league_runs = 4.50

    off_h = np.clip(float(ops_loc) / league_ops, 0.75, 1.30)
    off_a = np.clip(float(ops_vis) / league_ops, 0.75, 1.30)
    sp_h = np.clip(float(pitcher_loc_era) / league_era, 0.65, 1.45)
    sp_a = np.clip(float(pitcher_vis_era) / league_era, 0.65, 1.45)
    bp_h = np.clip(float(bullpen_loc_era) / league_era, 0.70, 1.40)
    bp_a = np.clip(float(bullpen_vis_era) / league_era, 0.70, 1.40)
    park = np.clip(float(park_factor) / 100.0, 0.88, 1.15)
    climate = calcular_factor_clima(viento_mph, direccion_viento, temp_f)

    # Aproximación de innings: abridor ~5.5, bullpen ~3.5. El bullpen debe ser
    # ERA de relevistas; si la app no dispone de uno real, debe marcarlo como tal.
    pit_a = (sp_a * 5.5 + bp_a * 3.5) / 9.0
    pit_h = (sp_h * 5.5 + bp_h * 3.5) / 9.0

    sab_h = league_runs * off_h * pit_a * park * climate * 1.025
    sab_a = league_runs * off_a * pit_h * park * climate

    # La forma reciente aporta señal, pero no domina a la proyección estructural.
    exp_h = 0.72 * sab_h + 0.28 * recent_h["rf"]
    exp_a = 0.72 * sab_a + 0.28 * recent_a["rf"]
    exp_h = float(np.clip(exp_h, 1.5, 8.5))
    exp_a = float(np.clip(exp_a, 1.5, 8.5))

    # Negative Binomial captura mejor la sobredispersión de carreras que Poisson.
    dispersion = 12.0
    p_h = dispersion / (dispersion + exp_h)
    p_a = dispersion / (dispersion + exp_a)
    rng = np.random.default_rng(42)
    rh = rng.negative_binomial(dispersion, p_h, int(num_simulaciones))
    ra = rng.negative_binomial(dispersion, p_a, int(num_simulaciones))

    ties = rh == ra
    if np.any(ties):
        # En extra innings hay ganador; pequeño home edge, sin fabricar carreras
        # para los cálculos de total previos.
        home_wins_tie = rng.random(np.sum(ties)) < 0.52
    else:
        home_wins_tie = np.array([], dtype=bool)
    home_win = rh > ra
    home_win[ties] = home_wins_tie

    totals = rh + ra
    diff = rh - ra
    line = float(linea_carreras_casino)
    over = float(np.mean(totals > line) * 100.0)
    under = float(np.mean(totals < line) * 100.0)
    push = float(np.mean(np.isclose(totals, line)) * 100.0) if line.is_integer() else 0.0

    carreras = {
        "Promedio_Total": round(float(np.mean(totals)), 2),
        f"Over {line}": round(over, 2),
        f"Under {line}": round(under, 2),
        f"Push {line}": round(push, 2),
    }
    if spread_loc is not None:
        p, pu = _runline_prob(diff, spread_loc, "home")
        carreras[f"Spread Local {float(spread_loc):+.1f}"] = round(p, 2)
        carreras[f"Push Spread Local {float(spread_loc):+.1f}"] = round(pu, 2)
    if spread_vis is not None:
        p, pu = _runline_prob(diff, spread_vis, "away")
        carreras[f"Spread Visita {float(spread_vis):+.1f}"] = round(p, 2)
        carreras[f"Push Spread Visita {float(spread_vis):+.1f}"] = round(pu, 2)

    p_home = float(np.mean(home_win) * 100.0)
    return {
        "Moneyline": {"Gana Local": round(p_home, 2), "Gana Visita": round(100.0 - p_home, 2)},
        "Carreras": carreras,
        "Expectativas": {h: round(exp_h, 3), a: round(exp_a, 3)},
        "Metadatos": {
            "OPS_Local": round(float(ops_loc), 3), "OPS_Visita": round(float(ops_vis), 3),
            "ERA_Abridor_Local": round(float(pitcher_loc_era), 2), "ERA_Abridor_Visita": round(float(pitcher_vis_era), 2),
            "ParkFactor": round(float(park_factor), 1), "ClimaFactor": round(climate, 3),
            "Altitud_ft": float(altitud_ft or 0), "Nota_Altitud": "capturada en park factor; no se duplica el ajuste",
        },
    }
