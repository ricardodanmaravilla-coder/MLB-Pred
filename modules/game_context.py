from __future__ import annotations

import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from .team_utils import normalize_team
from .multi_odds import install_requests_bridge

SLATE_TZ = ZoneInfo("America/New_York")
ROOF_OR_DOME_TEAMS = {"AZ", "HOU", "MIA", "MIL", "SEA", "TB", "TEX", "TOR"}


def _prime_primary_odds_key_from_streamlit():
    """Preserve an existing The Odds API key stored only in Streamlit Secrets."""
    if os.getenv("ODDS_API_KEY", "").strip():
        return
    try:
        import streamlit as st
        value = str(st.secrets.get("ODDS_API_KEY", "")).strip()
        if value:
            os.environ["ODDS_API_KEY"] = value
    except Exception:
        pass


# app_mlb imports this module before it resolves ODDS_API_KEY. Prime the legacy
# key first, then install the compatibility layer. This keeps the existing
# scanner/EV/Kelly pipeline untouched while allowing alternate providers.
_prime_primary_odds_key_from_streamlit()
install_requests_bridge()


def slate_date(now=None):
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
            'team': target, 'park_factor': float(row[pf_col]),
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


def match_odds_game(odds_games, mlb_game, max_hours=2.0):
    candidates = [g for g in (odds_games or []) if _same_matchup(g, mlb_game)]
    if not candidates:
        return None
    target = parse_utc(mlb_game.get('start_time_utc'))
    if target is None:
        if len(candidates) == 1:
            return candidates[0]
        # Multiple providers can legitimately describe the same game. Merge
        # their bookmakers rather than treating them as an ambiguous doubleheader.
        merged = dict(candidates[0])
        merged['bookmakers'] = []
        for c in candidates:
            merged['bookmakers'].extend(c.get('bookmakers', []) or [])
        return merged
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

    # Provider duplicates usually have effectively the same start time. Merge
    # those snapshots; only reject truly distinct games (e.g. a doubleheader).
    close = [(delta, game) for delta, game in scored if delta <= float(max_hours)]
    near_best = [(delta, game) for delta, game in close if abs(delta - scored[0][0]) < 0.25]
    if len(near_best) > 1:
        starts = [parse_utc(g.get('commence_time')) for _, g in near_best]
        starts = [s for s in starts if s is not None]
        if starts and (max(starts) - min(starts)).total_seconds() <= 30 * 60:
            merged = dict(near_best[0][1])
            merged['bookmakers'] = []
            for _, g in near_best:
                merged['bookmakers'].extend(g.get('bookmakers', []) or [])
            return merged
        return None
    return scored[0][1]


def market_from_event(event, american_to_decimal):
    """Choose one bookmaker and keep all markets internally coherent.

    Prefer the bookmaker with the most complete h2h/totals/spreads snapshot instead of
    taking the first h2h book and accidentally discarding usable totals or run lines.
    """
    empty = {
        'linea_carreras': None, 'cuota_loc': None, 'cuota_vis': None,
        'cuota_over': None, 'cuota_under': None,
        'spread_loc': None, 'cuota_spread_loc': None,
        'spread_vis': None, 'cuota_spread_vis': None,
        'bookmaker': None,
    }
    if not event:
        return dict(empty)
    home = event.get('home_team')
    away = event.get('away_team')
    snapshots = []
    for bookmaker in event.get('bookmakers', []):
        snap = dict(empty)
        found = set()
        for market in bookmaker.get('markets', []):
            key = market.get('key')
            outcomes = market.get('outcomes', [])
            if key == 'h2h':
                for o in outcomes:
                    name = normalize_team(o.get('name'))
                    if name == normalize_team(home):
                        snap['cuota_loc'] = american_to_decimal(o.get('price'))
                    elif name == normalize_team(away):
                        snap['cuota_vis'] = american_to_decimal(o.get('price'))
                if snap['cuota_loc'] and snap['cuota_vis']:
                    found.add('h2h')
            elif key == 'totals':
                over_point = under_point = None
                for o in outcomes:
                    if o.get('name') == 'Over':
                        over_point = o.get('point')
                        snap['cuota_over'] = american_to_decimal(o.get('price'))
                    elif o.get('name') == 'Under':
                        under_point = o.get('point')
                        snap['cuota_under'] = american_to_decimal(o.get('price'))
                if over_point is not None and under_point is not None and float(over_point) == float(under_point) and snap['cuota_over'] and snap['cuota_under']:
                    snap['linea_carreras'] = float(over_point)
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
                if (snap['spread_loc'] is not None and snap['spread_vis'] is not None
                        and abs(snap['spread_loc'] + snap['spread_vis']) < 1e-9
                        and snap['cuota_spread_loc'] and snap['cuota_spread_vis']):
                    found.add('spreads')
                else:
                    snap['spread_loc'] = snap['cuota_spread_loc'] = None
                    snap['spread_vis'] = snap['cuota_spread_vis'] = None
        if 'h2h' in found:
            snap['bookmaker'] = bookmaker.get('title') or bookmaker.get('key')
            snapshots.append((len(found), snap))
    if not snapshots:
        return dict(empty)
    snapshots.sort(key=lambda x: x[0], reverse=True)
    return snapshots[0][1]
