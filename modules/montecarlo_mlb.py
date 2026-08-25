import numpy as np
import pandas as pd

from .team_utils import normalize_team
from .historical_mlb import prepare_games


def calcular_factor_clima(viento_mph, direccion_viento, temp_f):
    try:
        wind = float(viento_mph or 0)
        temp = float(temp_f or 72)
    except (TypeError, ValueError):
        wind, temp = 0.0, 72.0

    d = str(direccion_viento or '').lower()
    mult = 1.0
    if 'outfield' in d or 'hacia afuera' in d:
        mult += min(wind, 25.0) * 0.004
    elif 'infield' in d or 'hacia adentro' in d:
        mult -= min(wind, 25.0) * 0.004
    mult += np.clip(temp - 72.0, -30.0, 30.0) * 0.0015
    return float(np.clip(mult, 0.88, 1.12))


def _games(df):
    g = prepare_games(df)
    if not g.empty:
        g = g.copy()
        g['HomeKey'] = g.Home.map(normalize_team)
        g['AwayKey'] = g.Away.map(normalize_team)
    return g


def obtener_h2h_detalle(df_games, loc_abbr, vis_abbr, n=12):
    """Historical H2H diagnostic only.

    V3 deliberately does not blend this back into Monte Carlo because H2H is
    already represented by the historical ML model. This prevents double count.
    """
    g = _games(df_games)
    if g.empty:
        return 50.0, 0
    loc, vis = normalize_team(loc_abbr), normalize_team(vis_abbr)
    rows = g[((g.HomeKey == loc) & (g.AwayKey == vis)) |
             ((g.HomeKey == vis) & (g.AwayKey == loc))].tail(n)
    wins = valid = 0
    for _, r in rows.iterrows():
        hs, aw = float(r.Home_Score), float(r.Away_Score)
        if hs == aw:
            continue
        valid += 1
        if (r.HomeKey == loc and hs > aw) or (r.AwayKey == loc and aw > hs):
            wins += 1
    if not valid:
        return 50.0, 0
    raw = wins / valid
    weight = valid / (valid + 10.0)
    shrunk = 0.5 + (raw - 0.5) * weight
    return float(shrunk * 100.0), valid


def obtener_h2h(df_games, loc_abbr, vis_abbr):
    return obtener_h2h_detalle(df_games, loc_abbr, vis_abbr)[0]


def obtener_carreras_recientes(df_games, equipo_abbr, n=10):
    """Kept for UI diagnostics/backward compatibility; not blended into MC V3."""
    g = _games(df_games)
    if g.empty:
        return None
    t = normalize_team(equipo_abbr)
    rows = g[(g.HomeKey == t) | (g.AwayKey == t)].tail(int(n))
    vals = [float(r.Home_Score if r.HomeKey == t else r.Away_Score) for _, r in rows.iterrows()]
    return float(np.mean(vals)) if vals else None


def _spread_probability(diff, side, line):
    line = float(line)
    if side == 'local':
        return float(np.mean((diff + line) > 0) * 100.0)
    return float(np.mean((-diff + line) > 0) * 100.0)


def _resolve_extra_innings(home, away, tied_mask, exp_home, exp_away, rng):
    """Resolve tied regulation simulations consistently for ML/totals/spreads.

    Previous code simply awarded one run to one team. V3 simulates extra innings
    with the automatic runner environment and keeps those extra runs in all markets.
    """
    idx = np.flatnonzero(tied_mask)
    if len(idx) == 0:
        return home, away, 0

    # Approximate per-extra-inning run environment. The ghost runner increases scoring;
    # keep the multiplier modest and derive team rates from the pregame run expectation.
    lam_h = max(0.25, min(1.25, (float(exp_home) / 9.0) * 1.35))
    lam_a = max(0.25, min(1.25, (float(exp_away) / 9.0) * 1.35))
    active = idx.copy()
    innings_used = 0

    for inning in range(1, 7):
        if len(active) == 0:
            break
        innings_used = inning
        add_a = rng.poisson(lam_a, len(active))
        add_h = rng.poisson(lam_h * 1.02, len(active))
        away[active] += add_a
        home[active] += add_h
        active = active[home[active] == away[active]]

    # Extremely rare unresolved ties after six extras: resolve with one additional
    # run but preserve the run in totals/spreads and only on the remaining simulations.
    if len(active):
        home_win = rng.random(len(active)) < 0.52
        home[active] += home_win.astype(int)
        away[active] += (~home_win).astype(int)

    return home, away, innings_used


def simular_partido_mlb(
    local,
    visita,
    pitcher_loc_xfip,
    pitcher_vis_xfip,
    wrc_loc,
    wrc_vis,
    bullpen_loc_era,
    bullpen_vis_era,
    park_factor,
    altitud_ft,
    viento_mph,
    direccion_viento,
    temp_f,
    linea_carreras_casino,
    df_games=None,
    num_simulaciones=50000,
):
    if linea_carreras_casino is None or float(linea_carreras_casino) <= 0:
        raise ValueError('Línea de carreras de casino requerida y no disponible.')

    vals = [wrc_loc, wrc_vis, pitcher_loc_xfip, pitcher_vis_xfip,
            bullpen_loc_era, bullpen_vis_era, park_factor]
    if any(v is None or pd.isna(v) for v in vals):
        raise ValueError(f'Datos incompletos para simular {visita} @ {local}.')

    loc, vis = normalize_team(local), normalize_team(visita)

    # Offensive index. Legacy wRC+ may actually be an OPS-based index, but it is
    # still centered around the historical scale used by this project.
    bat_l = np.clip(float(wrc_loc) / 100.0, 0.75, 1.25)
    bat_v = np.clip(float(wrc_vis) / 100.0, 0.75, 1.25)

    # The individual-pitcher file currently stores ERA in the legacy xFIP column.
    # Treat it explicitly as starter run prevention, not true xFIP.
    sp_l = np.clip(float(pitcher_loc_xfip) / 4.10, 0.70, 1.30)
    sp_v = np.clip(float(pitcher_vis_xfip) / 4.10, 0.70, 1.30)

    # Production does not yet have a true bullpen-only CSV. This input is therefore
    # team pitching ERA and gets a smaller weight so the starter is not double-counted.
    team_pitch_l = np.clip(float(bullpen_loc_era) / 4.10, 0.75, 1.30)
    team_pitch_v = np.clip(float(bullpen_vis_era) / 4.10, 0.75, 1.30)

    climate = calcular_factor_clima(viento_mph, direccion_viento, temp_f)
    park = np.clip(float(park_factor) / 100.0, 0.88, 1.12)
    try:
        altitude = 1.0 + float(np.clip((float(altitud_ft or 0) - 1000.0) / 1000.0 * 0.0015, 0.0, 0.008))
    except (TypeError, ValueError):
        altitude = 1.0

    # V3 contextual MC: no recent-form or H2H blending here. Those are owned by ML.
    # Starter receives 72% and aggregate team pitching 28% as a conservative proxy
    # until a real bullpen-only data source is persisted in production.
    opp_pitch_for_home = sp_v * 0.72 + team_pitch_v * 0.28
    opp_pitch_for_away = sp_l * 0.72 + team_pitch_l * 0.28

    exp_l = 4.55 * bat_l * opp_pitch_for_home * park * climate * altitude
    exp_v = 4.45 * bat_v * opp_pitch_for_away * park * climate * altitude
    exp_l = float(np.clip(exp_l, 1.5, 8.5))
    exp_v = float(np.clip(exp_v, 1.5, 8.5))

    sims = int(np.clip(num_simulaciones, 5000, 100000))
    rng = np.random.default_rng()
    dispersion = 14.5

    home = rng.negative_binomial(dispersion, dispersion / (dispersion + exp_l), sims)
    away = rng.negative_binomial(dispersion, dispersion / (dispersion + exp_v), sims)
    regulation_ties = home == away
    home, away, max_extra_innings = _resolve_extra_innings(home, away, regulation_ties, exp_l, exp_v, rng)

    prob_l = float(np.mean(home > away) * 100.0)
    prob_v = 100.0 - prob_l
    totals = home + away
    diff = home - away
    line = float(linea_carreras_casino)

    runs = {
        'Promedio_Total': round(float(np.mean(totals)), 2),
        f'Over {linea_carreras_casino}': round(float(np.mean(totals > line) * 100.0), 2),
        f'Under {linea_carreras_casino}': round(float(np.mean(totals < line) * 100.0), 2),
        f'Push {linea_carreras_casino}': round(float(np.mean(totals == line) * 100.0), 2),
    }
    for spread in np.arange(-5.5, 6.0, 0.5):
        if abs(spread) < 1e-9:
            continue
        spread = float(spread)
        runs[f'Spread Local {spread:+.1f}'] = round(_spread_probability(diff, 'local', spread), 2)
        runs[f'Spread Visita {spread:+.1f}'] = round(_spread_probability(diff, 'visita', spread), 2)

    h2h, h2h_n = obtener_h2h_detalle(df_games, loc, vis)
    pexp = (exp_l + exp_v) ** 0.285
    pyth = (exp_l ** pexp) / ((exp_l ** pexp) + (exp_v ** pexp)) * 100.0

    return {
        'Moneyline': {'Gana Local': round(prob_l, 2), 'Gana Visita': round(prob_v, 2)},
        'Carreras': runs,
        'Metadatos': {
            'Pythagenpat_Loc': round(float(pyth), 2),
            'H2H_Loc': round(h2h, 2),
            'H2H_Muestra': h2h_n,
            'H2H_Peso': 0.0,
            'H2H_Usado_En_Probabilidad': False,
            'Forma_Reciente_Usada_En_MC': False,
            'Factor_Clima': round(climate, 4),
            'Factor_Altitud_Residual': round(altitude, 4),
            'Carreras_Exp_Local': round(exp_l, 2),
            'Carreras_Exp_Visita': round(exp_v, 2),
            'Empates_Regulacion': int(regulation_ties.sum()),
            'Max_Extra_Innings_Simulados': int(max_extra_innings),
            'Pitching_Agregado_Es_Proxy_Bullpen': True,
            'Simulaciones': sims,
        },
    }
