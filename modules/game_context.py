from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from .team_utils import normalize_team

SLATE_TZ = ZoneInfo("America/New_York")
ROOF_OR_DOME_TEAMS = {"AZ", "HOU", "MIA", "MIL", "SEA", "TB", "TEX", "TOR"}


def slate_date(now=None):
    """Return the MLB slate date in US Eastern time, independent of host timezone."""
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(SLATE_TZ).date()


def parse_utc(value):
    if value in (None, ""):
        return None
    try:
        dt = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.to_pydatetime()
    except Exception:
        return None


def conservative_auto_weather(team, start_time_utc, temp_f, wind_mph, wind_dir, now=None):
    """Use current weather only near first pitch and never assume a retractable roof is open.

    If the roof state is unknown or first pitch is more than two hours away, return
    neutral inputs rather than injecting current conditions as if they were a forecast.
    """
    target = normalize_team(team)
    if target in ROOF_OR_DOME_TEAMS:
        return 72, 0, "None", "neutral_roof_unknown"
    start = parse_utc(start_time_utc)
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    if start is None or abs((start - current).total_seconds()) > 2 * 3600:
        return 72, 0, "None", "neutral_not_near_first_pitch"
    if temp_f is None or wind_mph is None:
        return 72, 0, "None", "neutral_weather_unavailable"
    return temp_f, wind_mph, wind_dir or "None", "current_near_first_pitch"


def park_for_team(df_parks, team):
    """Resolve a park row using the canonical team normalizer instead of raw aliases."""
    if df_parks is None or df_parks.empty:
        return None
    cols = list(df_parks.columns)
    team_col = next((c for c in ['Team', 'TeamCode', 'Abbr', 'Franchise', 'Equipo', 'franchise'] if c in cols), None)
    pf_col = next((c for c in cols if 'park_factor' in c.lower() or c.lower() in ('factor', 'pf')), None)
    alt_col = next((c for c in cols if 'altitud' in c.lower() or 'elevation' in c.lower() or c.lower() in ('alt', 'altitude')), None)
    if team_col is None or pf_col is None or alt_col is None:
        return None
    target = normalize_team(team)
    keys = df_parks[team_col].map(normalize_team)
    rows = df_parks[keys == target]
    if rows.empty:
        return None
    row = rows.iloc[-1]
    try:
        return {
            'team': target,
            'park_factor': float(row[pf_col]),
            'altitude_ft': float(row[alt_col]),
            'stadium': row.get('Estadio', row.get('Stadium', 'Unknown')),
        }
    except (TypeError, ValueError):
        return None


def _same_matchup(odds_game, mlb_game):
    return (
        normalize_team(odds_game.get('home_team')) == normalize_team(mlb_game.get('local'))
        and normalize_team(odds_game.get('away_team')) == normalize_team(mlb_game.get('visita'))
    )


def match_odds_game(odds_games, mlb_game, max_hours=6.0):
    """Match odds to an MLB game by both teams and start time.

    This prevents doubleheaders from sharing the same sportsbook event merely because
    they have the same home team. If start times are unavailable or ambiguous, return
    None instead of attaching potentially wrong prices.
    """
    candidates = [g for g in (odds_games or []) if _same_matchup(g, mlb_game)]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    target = parse_utc(mlb_game.get('start_time_utc'))
    if target is None:
        return None
    scored = []
    for g in candidates:
        dt = parse_utc(g.get('commence_time'))
        if dt is None:
            continue
        delta = abs((dt - target).total_seconds()) / 3600.0
        scored.append((delta, g))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0])
    if scored[0][0] > float(max_hours):
        return None
    if len(scored) > 1 and abs(scored[1][0] - scored[0][0]) < 0.25:
        return None
    return scored[0][1]


def market_from_event(event, american_to_decimal):
    """Extract one coherent bookmaker snapshot from a matched odds event."""
    out = {
        'linea_carreras': None, 'cuota_loc': None, 'cuota_vis': None,
        'cuota_over': None, 'cuota_under': None,
        'spread_loc': None, 'cuota_spread_loc': None,
        'spread_vis': None, 'cuota_spread_vis': None,
        'bookmaker': None,
    }
    if not event:
        return out
    home = event.get('home_team')
    away = event.get('away_team')
    for bookmaker in event.get('bookmakers', []):
        found = set()
        snap = dict(out)
        for market in bookmaker.get('markets', []):
            key = market.get('key')
            outcomes = market.get('outcomes', [])
            if key == 'h2h':
                for o in outcomes:
                    if normalize_team(o.get('name')) == normalize_team(home):
                        snap['cuota_loc'] = american_to_decimal(o.get('price'))
                    elif normalize_team(o.get('name')) == normalize_team(away):
                        snap['cuota_vis'] = american_to_decimal(o.get('price'))
                if snap['cuota_loc'] and snap['cuota_vis']:
                    found.add('h2h')
            elif key == 'totals':
                points = set()
                for o in outcomes:
                    if o.get('point') is not None:
                        points.add(float(o['point']))
                    if o.get('name') == 'Over':
                        snap['cuota_over'] = american_to_decimal(o.get('price'))
                    elif o.get('name') == 'Under':
                        snap['cuota_under'] = american_to_decimal(o.get('price'))
                if len(points) == 1 and snap['cuota_over'] and snap['cuota_under']:
                    snap['linea_carreras'] = points.pop()
                    found.add('totals')
            elif key == 'spreads':
                for o in outcomes:
                    name = normalize_team(o.get('name'))
                    point = o.get('point')
                    if name == normalize_team(home) and point is not None:
                        snap['spread_loc'] = float(point)
                        snap['cuota_spread_loc'] = american_to_decimal(o.get('price'))
                    elif name == normalize_team(away) and point is not None:
                        snap['spread_vis'] = float(point)
                        snap['cuota_spread_vis'] = american_to_decimal(o.get('price'))
                if snap['cuota_spread_loc'] and snap['cuota_spread_vis']:
                    found.add('spreads')
        if 'h2h' in found:
            snap['bookmaker'] = bookmaker.get('title') or bookmaker.get('key')
            return snap
    return out
