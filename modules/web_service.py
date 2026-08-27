"""Reusable service layer for the Cloud Run web application.

This module deliberately does not import Streamlit. It uses the same production
V7 ML, Monte Carlo and scanner engines as app_mlb.py so the Cloud Run frontend
cannot drift into a second prediction model.
"""
from __future__ import annotations

import math
import os
import threading
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .game_context import (
    conservative_auto_weather,
    market_from_event,
    match_odds_game,
    park_for_team,
    slate_date,
)
from .metric_quality import batting_metric, pitching_metric, row_pitching_value
from .ml_mlb import PredictorMLMLB
from .montecarlo_mlb import simular_partido_mlb
from .pick_ledger import append_snapshot, persistent_backend_available
from .scanner_engine import moneyline_candidate, no_vig_two_way, runline_candidate, total_candidate

DATA_DIR = Path(os.getenv("MLB_DATA_DIR", "data"))
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()

EQUIPOS_MAP = {
    "New York Yankees": "NYY", "Boston Red Sox": "BOS", "Los Angeles Dodgers": "LAD",
    "Houston Astros": "HOU", "Atlanta Braves": "ATL", "Philadelphia Phillies": "PHI",
    "Baltimore Orioles": "BAL", "Tampa Bay Rays": "TB", "Toronto Blue Jays": "TOR",
    "Chicago White Sox": "CWS", "Cleveland Guardians": "CLE", "Detroit Tigers": "DET",
    "Kansas City Royals": "KC", "Minnesota Twins": "MIN", "Los Angeles Angels": "LAA",
    "Oakland Athletics": "OAK", "Athletics": "OAK", "Sacramento Athletics": "OAK",
    "Seattle Mariners": "SEA", "Texas Rangers": "TEX", "Chicago Cubs": "CHC",
    "Cincinnati Reds": "CIN", "Milwaukee Brewers": "MIL", "Pittsburgh Pirates": "PIT",
    "St. Louis Cardinals": "STL", "Arizona Diamondbacks": "AZ", "Colorado Rockies": "COL",
    "San Francisco Giants": "SF", "San Diego Padres": "SD", "Miami Marlins": "MIA",
    "New York Mets": "NYM", "Washington Nationals": "WSH",
}

CITIES = {
    "New York Yankees": "New_York", "Boston Red Sox": "Boston", "Los Angeles Dodgers": "Los_Angeles",
    "Houston Astros": "Houston", "Atlanta Braves": "Atlanta", "Philadelphia Phillies": "Philadelphia",
    "Baltimore Orioles": "Baltimore", "Tampa Bay Rays": "St_Petersburg", "Toronto Blue Jays": "Toronto",
    "Chicago White Sox": "Chicago", "Cleveland Guardians": "Cleveland", "Detroit Tigers": "Detroit",
    "Kansas City Royals": "Kansas_City", "Minnesota Twins": "Minneapolis", "Los Angeles Angels": "Anaheim",
    "Oakland Athletics": "Oakland", "Athletics": "Oakland", "Sacramento Athletics": "Sacramento",
    "Seattle Mariners": "Seattle", "Texas Rangers": "Arlington", "Chicago Cubs": "Chicago",
    "Cincinnati Reds": "Cincinnati", "Milwaukee Brewers": "Milwaukee", "Pittsburgh Pirates": "Pittsburgh",
    "St. Louis Cardinals": "St_Louis", "Arizona Diamondbacks": "Phoenix", "Colorado Rockies": "Denver",
    "San Francisco Giants": "San_Francisco", "San Diego Padres": "San_Diego", "Miami Marlins": "Miami",
    "New York Mets": "New_York", "Washington Nationals": "Washington",
}


def american_to_decimal(value):
    try:
        x = float(value)
        if x == 0:
            return None
        return round((x / 100.0) + 1.0, 2) if x > 0 else round((100.0 / abs(x)) + 1.0, 2)
    except (TypeError, ValueError):
        return None


def estimate_ml_probability(projection, line, kind="over", sigma=None):
    if projection is None or line is None:
        return 50.0
    try:
        sigma = max(1.0, float(sigma if sigma is not None else (3.5 if kind in ("over", "under") else 4.2)))
        if kind == "over":
            z = (float(projection) - float(line)) / sigma
        elif kind == "under":
            z = (float(line) - float(projection)) / sigma
        else:
            z = (float(projection) + float(line)) / sigma
        return round(max(0.0, min(100.0, 50.0 * (1.0 + math.erf(z / math.sqrt(2.0))))), 2)
    except Exception:
        return 50.0


def kelly_fraction_pct(probability, decimal_odds, fraction=0.25, push_probability=0.0):
    try:
        p = max(0.0, float(probability) / 100.0)
        push = max(0.0, float(push_probability) / 100.0)
        q = max(0.0, 1.0 - p - push)
        b = float(decimal_odds) - 1.0
        decisions = p + q
        if b <= 0 or decisions <= 0:
            return 0.0
        k = (b * p - q) / (b * decisions)
        return round(max(0.0, k * float(fraction)) * 100.0, 2)
    except Exception:
        return 0.0


class MLBWebService:
    def __init__(self):
        self._lock = threading.RLock()
        self._load_data_and_model()

    @staticmethod
    def _read_csv(name):
        path = DATA_DIR / name
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path, sep=None, engine="python", on_bad_lines="skip")
        df.columns = df.columns.str.strip()
        return df

    def _load_data_and_model(self):
        self.batting = self._read_csv("mlb_batting.csv")
        self.pitching = self._read_csv("mlb_pitching.csv")
        self.pitchers = self._read_csv("mlb_pitching_individual.csv")
        self.parks = self._read_csv("mlb_park_factors.csv")
        self.games = self._read_csv("mlb_games.csv")
        self.bullpen = self._read_csv("mlb_bullpen.csv")
        for df in (self.batting, self.pitching, self.pitchers, self.bullpen):
            if 'Team' in df.columns:
                df['Team'] = df['Team'].astype(str).str.upper()
        self.predictor = PredictorMLMLB()
        self.model_ready = bool(
            not self.games.empty and not self.batting.empty and not self.pitching.empty
            and self.predictor.entrenar(self.batting, self.pitching, self.games)
        )

    def reload(self):
        with self._lock:
            self._load_data_and_model()
        return self.health()

    def health(self):
        return {
            "status": "ok" if self.model_ready else "degraded",
            "service": "mlb-pred-cloud-run",
            "model_ready": self.model_ready,
            "training_source": getattr(self.predictor, "training_source", None),
            "training_rows": int(getattr(self.predictor, "training_rows", 0) or 0),
            "signal_set": getattr(self.predictor, "signal_set", "baseline20"),
            "odds_configured": bool(ODDS_API_KEY),
            "ledger_persistent": bool(persistent_backend_available()),
            "slate_date": slate_date().isoformat(),
        }

    def _current_offensive_index(self, team, fallback=100.0):
        df = self.batting
        try:
            col = batting_metric(df)
            if not col or df.empty:
                return float(fallback)
            x = df.copy(); x['_v'] = pd.to_numeric(x[col], errors='coerce')
            if 'Season' in x.columns:
                x['_s'] = pd.to_numeric(x['Season'], errors='coerce')
                latest = x['_s'].dropna().max(); season = x[x['_s'] == latest]
            else:
                season = x
            rows = season[season['Team'] == team]
            if rows.empty:
                rows = x[x['Team'] == team]
            if rows.empty:
                return float(fallback)
            value = float(rows['_v'].dropna().iloc[-1])
            if col == 'wRC+' and 'wRC+_Source' in rows.columns and 'FANGRAPHS_REAL' in str(rows.iloc[-1].get('wRC+_Source', '')):
                return float(np.clip(value, 70, 130))
            center = float(season['_v'].dropna().median())
            return float(np.clip((value / center) * 100.0, 75.0, 125.0)) if center else float(fallback)
        except Exception:
            return float(fallback)

    @staticmethod
    def _prior_stat(df, team, col, fallback, game_date=None):
        try:
            x = df[df['Team'] == team].copy()
            if x.empty or col not in x.columns:
                return float(fallback)
            if 'Season' in x.columns:
                x['_s'] = pd.to_numeric(x['Season'], errors='coerce')
                year = pd.Timestamp(game_date).year if game_date is not None else slate_date().year
                eligible = x[x['_s'] <= year - 1].sort_values('_s')
                if not eligible.empty:
                    v = pd.to_numeric(pd.Series([eligible.iloc[-1][col]]), errors='coerce').iloc[0]
                    if pd.notna(v): return float(v)
            v = pd.to_numeric(pd.Series([x.iloc[-1][col]]), errors='coerce').iloc[0]
            return float(v) if pd.notna(v) else float(fallback)
        except Exception:
            return float(fallback)

    def _starter_metric(self, name):
        try:
            if not name or name == "Por Anunciar" or self.pitchers.empty or 'Name' not in self.pitchers.columns:
                return None
            names = self.pitchers['Name'].astype(str)
            match = self.pitchers[names.str.casefold() == str(name).casefold()]
            if match.empty:
                last = str(name).split()[-1].casefold()
                fallback = self.pitchers[names.str.split().str[-1].str.casefold() == last]
                if fallback['Name'].nunique() != 1:
                    return None
                match = fallback
            value, _ = row_pitching_value(match.iloc[-1], None)
            return None if value is None else float(value)
        except Exception:
            return None

    def _team_pitching(self, team):
        rows = self.pitching[self.pitching['Team'] == team]
        if rows.empty:
            return None
        value, _ = row_pitching_value(rows.iloc[-1], None)
        return None if value is None else float(value)

    def _bullpen_metric(self, team, fallback=4.1):
        try:
            rows = self.bullpen[self.bullpen['Team'] == team]
            if rows.empty:
                return float(fallback)
            if 'Season' in rows.columns:
                rows = rows.assign(_s=pd.to_numeric(rows['Season'], errors='coerce')).sort_values('_s')
            value, _ = row_pitching_value(rows.iloc[-1], fallback)
            return float(value)
        except Exception:
            return float(fallback)

    def _weather(self, team, start_time):
        city = CITIES.get(team)
        temp = wind = None; direction = "None"
        if city:
            try:
                r = requests.get(f"https://wttr.in/{city}?format=j1", timeout=4)
                if r.ok:
                    curr = r.json().get('current_condition', [{}])[0]
                    temp = int(curr.get('temp_F')); wind = int(curr.get('windspeedMiles'))
                    compass = curr.get('winddir16Point', '')
                    direction = f"Compass {compass}" if compass else "None"
            except Exception:
                pass
        return conservative_auto_weather(team, start_time, temp, wind, direction)

    def slate(self):
        date = slate_date().strftime('%Y-%m-%d')
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}&hydrate=probablePitcher,team"
        games = []
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        for date_item in r.json().get('dates', []):
            for game in date_item.get('games', []):
                teams = game.get('teams', {})
                home = teams.get('home', {}).get('team', {}).get('name', '')
                away = teams.get('away', {}).get('team', {}).get('name', '')
                if not home or not away:
                    continue
                games.append({
                    "game_pk": game.get('gamePk'), "start_time_utc": game.get('gameDate'),
                    "home": home, "away": away,
                    "home_pitcher": teams.get('home', {}).get('probablePitcher', {}).get('fullName', 'Por Anunciar'),
                    "away_pitcher": teams.get('away', {}).get('probablePitcher', {}).get('fullName', 'Por Anunciar'),
                    "linea_carreras": None, "cuota_loc": None, "cuota_vis": None,
                    "cuota_over": None, "cuota_under": None,
                    "spread_loc": None, "cuota_spread_loc": None,
                    "spread_vis": None, "cuota_spread_vis": None,
                })
        if ODDS_API_KEY and games:
            odds_url = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/?apiKey={ODDS_API_KEY}&regions=us&markets=h2h,totals,spreads&oddsFormat=american"
            try:
                ro = requests.get(odds_url, timeout=8)
                if ro.ok:
                    events = ro.json()
                    for game in games:
                        event = match_odds_game(events, {"local": game['home'], "visita": game['away'], "game_pk": game['game_pk'], "start_time_utc": game['start_time_utc']})
                        if event is not None:
                            game.update(market_from_event(event, american_to_decimal))
            except Exception:
                pass
        return games

    def _evaluate_game(self, game):
        home_name, away_name = game['home'], game['away']
        h, a = EQUIPOS_MAP.get(home_name, ''), EQUIPOS_MAP.get(away_name, '')
        if not h or not a:
            raise ValueError("Equipo no normalizable")
        if game.get('cuota_loc') is None or game.get('cuota_vis') is None or game.get('linea_carreras') is None:
            raise ValueError("Cuotas/total no disponibles")

        off_h = self._current_offensive_index(h); off_a = self._current_offensive_index(a)
        pit_h = self._starter_metric(game.get('home_pitcher')) or self._team_pitching(h)
        pit_a = self._starter_metric(game.get('away_pitcher')) or self._team_pitching(a)
        if pit_h is None or pit_a is None:
            raise ValueError("Pitching insuficiente")
        bull_h = self._bullpen_metric(h, pit_h); bull_a = self._bullpen_metric(a, pit_a)
        park = park_for_team(self.parks, h)
        if not park:
            raise ValueError("Parque no resoluble")
        temp, wind, wind_dir, weather_source = self._weather(home_name, game.get('start_time_utc'))
        line = float(game['linea_carreras'])

        mc = simular_partido_mlb(
            local=home_name, visita=away_name,
            pitcher_loc_xfip=pit_h, pitcher_vis_xfip=pit_a,
            wrc_loc=off_h, wrc_vis=off_a,
            bullpen_loc_era=bull_h, bullpen_vis_era=bull_a,
            park_factor=park['park_factor'], altitud_ft=park['altitude_ft'],
            viento_mph=wind, direccion_viento=wind_dir, temp_f=temp,
            linea_carreras_casino=line, df_games=self.games, num_simulaciones=50000,
        )
        bat_col = batting_metric(self.batting) or ('OPS_Index' if 'OPS_Index' in self.batting.columns else 'wRC+')
        pit_col = pitching_metric(self.pitching) or ('ERA' if 'ERA' in self.pitching.columns else 'xFIP')
        game_date = slate_date()
        ml = self.predictor.predecir_partido(
            h, a,
            self._prior_stat(self.batting, h, bat_col, off_h, game_date),
            self._prior_stat(self.batting, a, bat_col, off_a, game_date),
            self._prior_stat(self.pitching, h, pit_col, bull_h, game_date),
            self._prior_stat(self.pitching, a, pit_col, bull_a, game_date),
            park['park_factor'], game_date=game_date,
        )
        runs = mc.get('Carreras', {})
        p_ml_h, p_ml_a = ml['Probabilidad_Local'], ml['Probabilidad_Visita']
        p_mc_h, p_mc_a = mc['Moneyline']['Gana Local'], mc['Moneyline']['Gana Visita']
        p_ml_over = estimate_ml_probability(ml.get('Proyeccion_Carreras', line), line, 'over', ml.get('Sigma_Carreras'))
        p_ml_under = estimate_ml_probability(ml.get('Proyeccion_Carreras', line), line, 'under', ml.get('Sigma_Carreras'))
        p_mc_over, p_mc_under = runs.get(f"Over {line}", 50.0), runs.get(f"Under {line}", 50.0)
        spread_h, spread_a = game.get('spread_loc'), game.get('spread_vis')
        p_ml_sp_h = estimate_ml_probability(ml.get('Proyeccion_Handicap_Local', 0), spread_h, 'spread', ml.get('Sigma_Handicap')) if spread_h is not None else 50.0
        p_ml_sp_a = estimate_ml_probability(-ml.get('Proyeccion_Handicap_Local', 0), spread_a, 'spread', ml.get('Sigma_Handicap')) if spread_a is not None else 50.0
        p_mc_sp_h = runs.get(f"Spread Local {float(spread_h):+.1f}", 50.0) if spread_h is not None else 50.0
        p_mc_sp_a = runs.get(f"Spread Visita {float(spread_a):+.1f}", 50.0) if spread_a is not None else 50.0

        nv_h, nv_a = no_vig_two_way(game.get('cuota_loc'), game.get('cuota_vis'))
        nv_over, nv_under = no_vig_two_way(game.get('cuota_over'), game.get('cuota_under'))
        nv_sp_h, nv_sp_a = no_vig_two_way(game.get('cuota_spread_loc'), game.get('cuota_spread_vis'))
        candidates = [
            (moneyline_candidate(f"Gana Local ({home_name})", p_ml_h, p_mc_h, game.get('cuota_loc'), nv_h), None),
            (moneyline_candidate(f"Gana Visita ({away_name})", p_ml_a, p_mc_a, game.get('cuota_vis'), nv_a), None),
        ]
        if game.get('cuota_over') is not None:
            candidates.append((total_candidate(f"Over {line}", p_ml_over, p_mc_over, game.get('cuota_over'), nv_over, runs.get(f"Push {line}", 0.0)), line))
        if game.get('cuota_under') is not None:
            candidates.append((total_candidate(f"Under {line}", p_ml_under, p_mc_under, game.get('cuota_under'), nv_under, runs.get(f"Push {line}", 0.0)), line))
        if spread_h is not None and game.get('cuota_spread_loc') is not None:
            candidates.append((runline_candidate(f"Hándicap {float(spread_h):+.1f} ({home_name})", p_ml_sp_h, p_mc_sp_h, game.get('cuota_spread_loc'), nv_sp_h, runs.get(f"Push Spread Local {float(spread_h):+.1f}", 0.0)), float(spread_h)))
        if spread_a is not None and game.get('cuota_spread_vis') is not None:
            candidates.append((runline_candidate(f"Hándicap {float(spread_a):+.1f} ({away_name})", p_ml_sp_a, p_mc_sp_a, game.get('cuota_spread_vis'), nv_sp_a, runs.get(f"Push Spread Visita {float(spread_a):+.1f}", 0.0)), float(spread_a)))

        accepted = []
        diagnostics = []
        for cand, market_line in candidates:
            if cand is None:
                continue
            row = {
                "game_pk": game.get('game_pk'), "partido": f"{away_name} @ {home_name}",
                "mercado": cand.market, "apuesta": cand.selection, "linea": market_line,
                "prob_ml": round(cand.prob_ml, 2), "prob_mc": round(cand.prob_mc, 2),
                "probabilidad": round(cand.probability, 2), "cuota": cand.odds,
                "no_vig": cand.market_no_vig, "edge_pp": cand.edge_pp, "ev_pct": cand.ev_pct,
                "desacuerdo_pp": cand.disagreement_pp, "score": cand.score,
                "push_probability": cand.push_probability,
                "kelly_pct": kelly_fraction_pct(cand.probability, cand.odds, push_probability=cand.push_probability),
                "accepted": bool(cand.accepted), "reason": cand.reason,
            }
            diagnostics.append(row)
            if cand.accepted:
                accepted.append(row)
        return {
            "game": game, "ml": ml,
            "monte_carlo": {"moneyline": mc.get('Moneyline', {}), "runs": runs},
            "context": {"park_factor": park['park_factor'], "altitude_ft": park['altitude_ft'], "temperature_f": temp, "wind_mph": wind, "wind_direction": wind_dir, "weather_source": weather_source},
            "accepted": accepted, "diagnostics": diagnostics,
        }

    def scan(self, persist=True):
        if not self.model_ready:
            raise RuntimeError("Modelo ML no disponible")
        games = self.slate(); accepted = []; diagnostics = []; errors = []
        for game in games:
            try:
                result = self._evaluate_game(game)
                accepted.extend(result['accepted']); diagnostics.extend(result['diagnostics'])
            except Exception as exc:
                errors.append({"game_pk": game.get('game_pk'), "partido": f"{game.get('away')} @ {game.get('home')}", "error": str(exc)[:200]})
        accepted = sorted(accepted, key=lambda r: float(r.get('score', -999)), reverse=True)[:3]
        if persist and accepted:
            ledger_rows = []
            for r in accepted:
                ledger_rows.append({
                    'game_date': slate_date().isoformat(), 'game_pk': r['game_pk'],
                    'away': r['partido'].split(' @ ')[0], 'home': r['partido'].split(' @ ')[1],
                    'market': r['mercado'], 'selection': r['apuesta'], 'line': r['linea'], 'odds': r['cuota'],
                    'prob_ml': r['prob_ml'], 'prob_mc': r['prob_mc'], 'prob_combined': r['probabilidad'],
                    'market_no_vig': r['no_vig'], 'edge_pp': r['edge_pp'], 'ev_pct': r['ev_pct'],
                    'disagreement_pp': r['desacuerdo_pp'], 'score': r['score'],
                    'model_version': 'v7-cloudrun', 'result_status': 'pending',
                })
            try:
                append_snapshot(ledger_rows)
            except Exception:
                pass
        return {
            "date": slate_date().isoformat(), "recommendations": accepted,
            "diagnostics": sorted(diagnostics, key=lambda r: float(r.get('score', -999)), reverse=True)[:12],
            "errors": errors, "games_seen": len(games), "model": self.health(),
        }
