import datetime
import os

import numpy as np
import pandas as pd
import requests
import streamlit as st

from modules.ml_mlb import PredictorMLMLB
from modules.montecarlo_mlb import simular_partido_mlb
from modules.odds_mlb import american_to_decimal, evaluar_dos_vias
from modules.team_utils import normalize_team

st.set_page_config(page_title="MLB Quant Analytics V2", layout="wide", page_icon="⚾")
st.title("⚾ MLB Quant Analytics — Model V2")
st.caption("MLB StatsAPI · OPS/ERA reales · forma prepartido · Monte Carlo · ML · mercado no-vig")

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
SEASON = datetime.date.today().year


@st.cache_data(ttl=3600)
def load_data():
    bat = pd.read_csv("data/mlb_batting.csv")
    pit = pd.read_csv("data/mlb_pitching.csv")
    games = pd.read_csv("data/mlb_games.csv")
    starters = pd.read_csv("data/mlb_pitching_individual.csv") if os.path.exists("data/mlb_pitching_individual.csv") else pd.DataFrame()
    parks = pd.read_csv("data/mlb_park_factors.csv")
    bullpen = pd.read_csv("data/mlb_bullpen.csv") if os.path.exists("data/mlb_bullpen.csv") else pd.DataFrame()
    for df in (bat, pit, starters, parks, bullpen):
        if not df.empty and "Team" in df.columns:
            df["Team"] = df.Team.map(normalize_team)
    return bat, pit, games, starters, parks, bullpen


@st.cache_data(ttl=300)
def schedule_today():
    day = datetime.date.today().isoformat()
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={day}&hydrate=probablePitcher,team"
    r = requests.get(url, timeout=12)
    r.raise_for_status()
    rows = []
    for block in r.json().get("dates", []):
        for game in block.get("games", []):
            home_obj = game.get("teams", {}).get("home", {})
            away_obj = game.get("teams", {}).get("away", {})
            home_name = home_obj.get("team", {}).get("name")
            away_name = away_obj.get("team", {}).get("name")
            h = normalize_team(home_name)
            a = normalize_team(away_name)
            if not h or not a:
                continue
            rows.append({
                "game_id": game.get("gamePk"), "home_name": home_name, "away_name": away_name,
                "home": h, "away": a,
                "home_pitcher": home_obj.get("probablePitcher", {}).get("fullName"),
                "away_pitcher": away_obj.get("probablePitcher", {}).get("fullName"),
                "game_date": game.get("gameDate"),
            })
    return rows


@st.cache_data(ttl=300)
def odds_today():
    if not ODDS_API_KEY:
        return []
    url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"
    params = {"apiKey": ODDS_API_KEY, "regions": "us", "markets": "h2h,totals,spreads", "oddsFormat": "american"}
    r = requests.get(url, params=params, timeout=12)
    if r.status_code != 200:
        return []
    return r.json()


def find_odds(game, odds):
    matches = []
    for item in odds:
        if normalize_team(item.get("home_team")) == game["home"] and normalize_team(item.get("away_team")) == game["away"]:
            matches.append(item)
    if not matches:
        return {}
    item = matches[0]
    # Se elige un bookmaker único por snapshot; nunca se mezclan lados de casas distintas.
    for book in item.get("bookmakers", []):
        out = {"bookmaker": book.get("title", book.get("key", "N/A"))}
        for market in book.get("markets", []):
            key = market.get("key")
            outcomes = market.get("outcomes", [])
            if key == "h2h":
                for o in outcomes:
                    team = normalize_team(o.get("name"))
                    if team == game["home"]: out["ml_home"] = american_to_decimal(o.get("price"))
                    if team == game["away"]: out["ml_away"] = american_to_decimal(o.get("price"))
            elif key == "totals":
                over = next((o for o in outcomes if o.get("name") == "Over"), None)
                under = next((o for o in outcomes if o.get("name") == "Under"), None)
                if over and under and over.get("point") == under.get("point"):
                    out["total_line"] = float(over["point"])
                    out["total_over"] = american_to_decimal(over.get("price"))
                    out["total_under"] = american_to_decimal(under.get("price"))
            elif key == "spreads":
                for o in outcomes:
                    team = normalize_team(o.get("name"))
                    if team == game["home"]:
                        out["spread_home"] = float(o.get("point")); out["spread_home_odd"] = american_to_decimal(o.get("price"))
                    if team == game["away"]:
                        out["spread_away"] = float(o.get("point")); out["spread_away_odd"] = american_to_decimal(o.get("price"))
        if out.get("ml_home") and out.get("ml_away"):
            return out
    return {}


def latest_team_row(df, team):
    rows = df[df.Team == normalize_team(team)].copy()
    if rows.empty:
        return None
    if "Season" in rows.columns:
        rows["Season"] = pd.to_numeric(rows.Season, errors="coerce")
        rows = rows.sort_values("Season")
    return rows.iloc[-1]


def team_ops(df_bat, team):
    r = latest_team_row(df_bat, team)
    if r is None:
        return None
    return float(pd.to_numeric(r.get("ops"), errors="coerce"))


def team_era(df_pit, team):
    r = latest_team_row(df_pit, team)
    if r is None:
        return None
    return float(pd.to_numeric(r.get("ERA", r.get("era")), errors="coerce"))


def starter_era(starters, name, team, fallback):
    if starters.empty or not name:
        return fallback, True
    rows = starters[(starters.Team == normalize_team(team)) & (starters.Name.astype(str).str.casefold() == str(name).casefold())]
    if rows.empty:
        return fallback, True
    return float(pd.to_numeric(rows.iloc[-1].ERA, errors="coerce")), False


def bullpen_era(bullpen, team, fallback):
    if bullpen.empty:
        return fallback, True
    rows = bullpen[bullpen.Team == normalize_team(team)]
    if rows.empty:
        return fallback, True
    return float(pd.to_numeric(rows.iloc[-1].Bullpen_ERA, errors="coerce")), False


def park_info(parks, team):
    rows = parks[parks.Team == normalize_team(team)]
    if rows.empty:
        return 100.0, 0.0
    return float(rows.iloc[-1].Park_Factor), float(rows.iloc[-1].Altitud)


def combine(a, b, wa=0.55):
    return wa * float(a) + (1.0-wa) * float(b)


bat, pit, games, starters, parks, bullpen = load_data()
model = PredictorMLMLB()
trained = model.entrenar(bat, pit, games)

st.sidebar.metric("Juegos históricos", f"{len(games):,}")
st.sidebar.write(f"ML entrenado: **{'Sí' if trained else 'No'}**")
st.sidebar.write(f"Odds API: **{'configurada' if ODDS_API_KEY else 'sin secret ODDS_API_KEY'}**")
st.sidebar.info("V2 no llama wRC+ a OPS ni xFIP a ERA. Si falta abridor/bullpen real, lo marca como proxy.")

try:
    games_today = schedule_today()
except Exception as exc:
    st.error(f"Error MLB StatsAPI: {exc}")
    games_today = []

all_odds = odds_today()
if not games_today:
    st.info("No hay juegos MLB programados para hoy.")
else:
    label = lambda g: f"{g['away_name']} @ {g['home_name']}"
    idx = st.selectbox("Partido", range(len(games_today)), format_func=lambda i: label(games_today[i]))
    game = games_today[idx]
    market = find_odds(game, all_odds)

    if st.button("Analizar partido V2", type="primary"):
        if not trained:
            st.error("ML no entrenó. Revisa datasets.")
            st.stop()
        ops_h, ops_a = team_ops(bat, game["home"]), team_ops(bat, game["away"])
        era_team_h, era_team_a = team_era(pit, game["home"]), team_era(pit, game["away"])
        if None in (ops_h, ops_a, era_team_h, era_team_a):
            st.error("Datos de equipo incompletos: NO BET")
            st.stop()
        sp_h, sp_h_proxy = starter_era(starters, game["home_pitcher"], game["home"], era_team_h)
        sp_a, sp_a_proxy = starter_era(starters, game["away_pitcher"], game["away"], era_team_a)
        bp_h, bp_h_proxy = bullpen_era(bullpen, game["home"], era_team_h)
        bp_a, bp_a_proxy = bullpen_era(bullpen, game["away"], era_team_a)
        pf, altitude = park_info(parks, game["home"])

        if not market.get("total_line"):
            st.warning("Sin línea real de total: se muestran ML/Monte Carlo, pero no se evalúa O/U.")
        line = market.get("total_line", 8.5)
        mc = simular_partido_mlb(
            game["home"], game["away"], ops_loc=ops_h, ops_vis=ops_a,
            pitcher_loc_era=sp_h, pitcher_vis_era=sp_a,
            bullpen_loc_era=bp_h, bullpen_vis_era=bp_a,
            park_factor=pf, altitud_ft=altitude,
            viento_mph=0, direccion_viento="", temp_f=72,
            linea_carreras_casino=line, df_games=games,
            spread_loc=market.get("spread_home"), spread_vis=market.get("spread_away"),
        )
        ml = model.predecir_partido(game["home"], game["away"], ops_h, ops_a, era_team_h, era_team_a, pf)
        ml["Prob_Over"] = model.prob_total(ml["Proyeccion_Carreras"], line, "over")
        ml["Prob_Under"] = model.prob_total(ml["Proyeccion_Carreras"], line, "under")

        st.subheader(label(game))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("OPS local", f"{ops_h:.3f}")
        c2.metric("OPS visita", f"{ops_a:.3f}")
        c3.metric("ERA abridor local", f"{sp_h:.2f}")
        c4.metric("ERA abridor visita", f"{sp_a:.2f}")
        st.caption(f"Abridores: {game['home_pitcher'] or 'TBA'} / {game['away_pitcher'] or 'TBA'} · Park factor {pf:.0f}")
        if sp_h_proxy or sp_a_proxy or bp_h_proxy or bp_a_proxy:
            st.warning("Hay datos proxy (abridor o bullpen). Se muestran probabilidades, pero V2 no debe tratarlas como pick de máxima confianza.")

        rows = [
            {"Mercado":"Local ML", "MC":mc["Moneyline"]["Gana Local"], "ML":ml["Probabilidad_Local"], "Ensemble":round(combine(ml["Probabilidad_Local"], mc["Moneyline"]["Gana Local"]),1)},
            {"Mercado":"Visitante ML", "MC":mc["Moneyline"]["Gana Visita"], "ML":ml["Probabilidad_Visita"], "Ensemble":round(combine(ml["Probabilidad_Visita"], mc["Moneyline"]["Gana Visita"]),1)},
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.write("Carreras esperadas Monte Carlo", mc["Expectativas"])
        st.metric("Total ML proyectado", ml["Proyeccion_Carreras"])

        diagnostics = []
        if market.get("ml_home") and market.get("ml_away"):
            ph = combine(ml["Probabilidad_Local"], mc["Moneyline"]["Gana Local"])
            pa = combine(ml["Probabilidad_Visita"], mc["Moneyline"]["Gana Visita"])
            diagnostics += evaluar_dos_vias("Moneyline Local", "Moneyline Visitante", ph, pa,
                                            market["ml_home"], market["ml_away"], min_prob=54, min_edge=3, min_ev=2)
        if market.get("total_over") and market.get("total_under") and ml.get("Prob_Over") is not None:
            po = combine(ml["Prob_Over"], mc["Carreras"][f"Over {float(line)}"])
            pu = combine(ml["Prob_Under"], mc["Carreras"][f"Under {float(line)}"])
            push = mc["Carreras"].get(f"Push {float(line)}", 0.0)
            diagnostics += evaluar_dos_vias(f"Over {line}", f"Under {line}", po, pu,
                                            market["total_over"], market["total_under"], push, push,
                                            min_prob=56, min_edge=4, min_ev=3)

        st.markdown("### Mercado y valor")
        if diagnostics:
            diag = pd.DataFrame(diagnostics)
            if sp_h_proxy or sp_a_proxy or bp_h_proxy or bp_a_proxy:
                diag.loc[diag.Estado == "VALUE BET", "Estado"] = "SEÑAL — DATOS PROXY"
            st.dataframe(diag, use_container_width=True, hide_index=True)
        else:
            st.info("Sin cuotas completas de una misma casa; no se calcula value bet.")
        with st.expander("Detalle técnico"):
            st.write("Bookmaker", market.get("bookmaker"))
            st.write("Cuotas", market)
            st.write("Monte Carlo", mc)
            st.write("ML", ml)
