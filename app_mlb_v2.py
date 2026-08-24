import datetime
import os
import traceback

import pandas as pd
import requests
import streamlit as st

from modules.ml_mlb import PredictorMLMLB
from modules.montecarlo_mlb import simular_partido_mlb
from modules.odds_mlb import american_to_decimal, evaluar_dos_vias
from modules.team_utils import normalize_team

st.set_page_config(page_title="MLB Quant Analytics V2", layout="wide", page_icon="⚾")
st.title("⚾ MLB Quant Analytics — Model V2")
st.caption("MLB StatsAPI · OPS/ERA reales · ML calibrado · Monte Carlo · mercado no-vig")

try:
    SECRET_ODDS = st.secrets.get("ODDS_API_KEY", "")
except Exception:
    SECRET_ODDS = ""
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "") or SECRET_ODDS


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
    out = []
    for block in r.json().get("dates", []):
        for game in block.get("games", []):
            hobj = game.get("teams", {}).get("home", {})
            aobj = game.get("teams", {}).get("away", {})
            hname = hobj.get("team", {}).get("name")
            aname = aobj.get("team", {}).get("name")
            h, a = normalize_team(hname), normalize_team(aname)
            if h and a:
                out.append({
                    "game_id": game.get("gamePk"), "home_name": hname, "away_name": aname,
                    "home": h, "away": a,
                    "home_pitcher": hobj.get("probablePitcher", {}).get("fullName"),
                    "away_pitcher": aobj.get("probablePitcher", {}).get("fullName"),
                    "game_date": game.get("gameDate"),
                })
    return out


@st.cache_data(ttl=300)
def odds_today():
    if not ODDS_API_KEY:
        return []
    r = requests.get(
        "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/",
        params={"apiKey": ODDS_API_KEY, "regions": "us", "markets": "h2h,totals,spreads", "oddsFormat": "american"},
        timeout=12,
    )
    return r.json() if r.status_code == 200 else []


def find_odds(game, odds):
    item = next((x for x in odds if normalize_team(x.get("home_team")) == game["home"] and normalize_team(x.get("away_team")) == game["away"]), None)
    if not item:
        return {}
    for book in item.get("bookmakers", []):
        out = {"bookmaker": book.get("title", book.get("key", "N/A"))}
        for market in book.get("markets", []):
            outcomes = market.get("outcomes", [])
            key = market.get("key")
            if key == "h2h":
                for o in outcomes:
                    t = normalize_team(o.get("name"))
                    if t == game["home"]:
                        out["ml_home"] = american_to_decimal(o.get("price"))
                    elif t == game["away"]:
                        out["ml_away"] = american_to_decimal(o.get("price"))
            elif key == "totals":
                ov = next((o for o in outcomes if o.get("name") == "Over"), None)
                un = next((o for o in outcomes if o.get("name") == "Under"), None)
                if ov and un and ov.get("point") == un.get("point"):
                    out.update(total_line=float(ov["point"]), total_over=american_to_decimal(ov.get("price")), total_under=american_to_decimal(un.get("price")))
            elif key == "spreads":
                for o in outcomes:
                    t = normalize_team(o.get("name"))
                    if t == game["home"]:
                        out.update(spread_home=float(o["point"]), spread_home_odd=american_to_decimal(o.get("price")))
                    elif t == game["away"]:
                        out.update(spread_away=float(o["point"]), spread_away_odd=american_to_decimal(o.get("price")))
        if out.get("ml_home") and out.get("ml_away"):
            return out
    return {}


def latest_row(df, team):
    rows = df[df.Team == normalize_team(team)].copy()
    if rows.empty:
        return None
    if "Season" in rows.columns:
        rows["Season"] = pd.to_numeric(rows.Season, errors="coerce")
        rows = rows.sort_values("Season")
    return rows.iloc[-1]


def get_num(row, *keys):
    if row is None:
        return None
    for key in keys:
        if key in row.index:
            x = pd.to_numeric(row.get(key), errors="coerce")
            if pd.notna(x):
                return float(x)
    return None


def starter_era(starters, name, team, fallback):
    if starters.empty or not name or "Name" not in starters.columns or "ERA" not in starters.columns:
        return fallback, True
    rows = starters[(starters.Team == normalize_team(team)) & (starters.Name.astype(str).str.casefold() == str(name).casefold())]
    if rows.empty:
        return fallback, True
    value = pd.to_numeric(rows.iloc[-1].ERA, errors="coerce")
    return (float(value), False) if pd.notna(value) else (fallback, True)


def bullpen_era(bullpen, team, fallback):
    if bullpen.empty or "Bullpen_ERA" not in bullpen.columns:
        return fallback, True
    rows = bullpen[bullpen.Team == normalize_team(team)]
    if rows.empty:
        return fallback, True
    value = pd.to_numeric(rows.iloc[-1].Bullpen_ERA, errors="coerce")
    return (float(value), False) if pd.notna(value) else (fallback, True)


def park_info(parks, team):
    rows = parks[parks.Team == normalize_team(team)]
    if rows.empty:
        return 100.0, 0.0
    pf = pd.to_numeric(rows.iloc[-1].get("Park_Factor"), errors="coerce")
    alt = pd.to_numeric(rows.iloc[-1].get("Altitud"), errors="coerce")
    return (float(pf) if pd.notna(pf) else 100.0, float(alt) if pd.notna(alt) else 0.0)


def blend(p_ml, p_mc):
    return 0.55 * float(p_ml) + 0.45 * float(p_mc)


def consensus_filter(rows, p_ml_a, p_mc_a, p_ml_b, p_mc_b, min_model=54.0, max_disagreement=10.0):
    pairs = [(p_ml_a, p_mc_a), (p_ml_b, p_mc_b)]
    for i, (pml, pmc) in enumerate(pairs):
        extra = []
        if min(float(pml), float(pmc)) < min_model:
            extra.append(f"modelo individual <{min_model:.0f}%")
        if abs(float(pml) - float(pmc)) > max_disagreement:
            extra.append(f"desacuerdo >{max_disagreement:.0f} pp")
        if extra and i < len(rows):
            rows[i]["Estado"] = "NO BET"
            rows[i]["Motivo"] = (rows[i].get("Motivo", "") + "; " + "; ".join(extra)).strip("; ")
    return rows


bat, pit, games, starters, parks, bullpen = load_data()
model = PredictorMLMLB()
trained = model.entrenar(bat, pit, games)
st.sidebar.metric("Juegos históricos", f"{len(games):,}")
st.sidebar.write(f"ML calibrado: **{'Sí' if trained and model.calibrado else 'No'}**")
st.sidebar.write(f"ODDS_API_KEY: **{'configurada' if ODDS_API_KEY else 'faltante'}**")
st.sidebar.info("Backtest OOS calibrado: 10,825 juegos. El modelo supera ligeramente el baseline; V2 usa filtros conservadores.")

try:
    slate = schedule_today()
except Exception as exc:
    st.error(f"MLB StatsAPI: {exc}")
    slate = []

try:
    market_feed = odds_today()
except Exception as exc:
    st.warning(f"The Odds API no disponible: {exc}")
    market_feed = []

if not slate:
    st.info("No hay juegos MLB programados para hoy.")
else:
    labels = [f"{g['away_name']} @ {g['home_name']}" for g in slate]
    choice = st.selectbox("Partido", range(len(slate)), format_func=lambda i: labels[i])
    game = slate[choice]
    market = find_odds(game, market_feed)

    if st.button("Analizar partido V2", type="primary"):
        try:
            if not trained:
                raise RuntimeError("El modelo no pudo entrenar")

            br_h, br_a = latest_row(bat, game["home"]), latest_row(bat, game["away"])
            pr_h, pr_a = latest_row(pit, game["home"]), latest_row(pit, game["away"])
            ops_h, ops_a = get_num(br_h, "ops"), get_num(br_a, "ops")
            era_h, era_a = get_num(pr_h, "ERA", "era"), get_num(pr_a, "ERA", "era")
            if None in (ops_h, ops_a, era_h, era_a):
                raise ValueError("Datos de equipo incompletos")

            sp_h, sp_h_proxy = starter_era(starters, game.get("home_pitcher"), game["home"], era_h)
            sp_a, sp_a_proxy = starter_era(starters, game.get("away_pitcher"), game["away"], era_a)
            bp_h, bp_h_proxy = bullpen_era(bullpen, game["home"], era_h)
            bp_a, bp_a_proxy = bullpen_era(bullpen, game["away"], era_a)
            pf, altitude = park_info(parks, game["home"])
            proxy = sp_h_proxy or sp_a_proxy or bp_h_proxy or bp_a_proxy
            line = float(market.get("total_line") or 8.5)

            with st.spinner("Calculando ML y Monte Carlo..."):
                mc = simular_partido_mlb(
                    game["home"], game["away"], ops_loc=ops_h, ops_vis=ops_a,
                    pitcher_loc_era=sp_h, pitcher_vis_era=sp_a,
                    bullpen_loc_era=bp_h, bullpen_vis_era=bp_a,
                    park_factor=pf, altitud_ft=altitude, viento_mph=0, direccion_viento="", temp_f=72,
                    linea_carreras_casino=line, df_games=games,
                    spread_loc=market.get("spread_home"), spread_vis=market.get("spread_away"),
                    num_simulaciones=50000,
                )
                ml = model.predecir_partido(game["home"], game["away"], ops_h, ops_a, era_h, era_a, pf)
                ml_over = model.prob_total(ml["Proyeccion_Carreras"], line, "over")
                ml_under = model.prob_total(ml["Proyeccion_Carreras"], line, "under")

            carreras = mc.get("Carreras", {})
            moneyline = mc.get("Moneyline", {})
            mc_home = moneyline.get("Gana Local")
            mc_away = moneyline.get("Gana Visita")
            mc_over = carreras.get(f"Over {float(line)}")
            mc_under = carreras.get(f"Under {float(line)}")
            if mc_home is None or mc_away is None:
                raise KeyError("Monte Carlo no devolvió probabilidades moneyline")

            st.subheader(labels[choice])
            a, b, c, d = st.columns(4)
            a.metric("OPS local", f"{ops_h:.3f}")
            b.metric("OPS visita", f"{ops_a:.3f}")
            c.metric("ERA abridor local", f"{sp_h:.2f}")
            d.metric("ERA abridor visita", f"{sp_a:.2f}")
            st.caption(f"Abridores: {game.get('home_pitcher') or 'TBA'} / {game.get('away_pitcher') or 'TBA'} · Park factor {pf:.0f} · Bookmaker {market.get('bookmaker','N/A')}")
            if proxy:
                st.warning("Falta al menos un dato real de abridor/bullpen. Probabilidades visibles, pero cualquier VALUE BET se degrada a SEÑAL — DATOS PROXY.")

            st.markdown("### Moneyline")
            money = pd.DataFrame([
                {"Lado": "Local", "ML": ml["Probabilidad_Local"], "MC": mc_home, "Ensemble": round(blend(ml["Probabilidad_Local"], mc_home), 1)},
                {"Lado": "Visitante", "ML": ml["Probabilidad_Visita"], "MC": mc_away, "Ensemble": round(blend(ml["Probabilidad_Visita"], mc_away), 1)},
            ])
            st.dataframe(money, use_container_width=True, hide_index=True)

            diagnostics = []
            if market.get("ml_home") and market.get("ml_away"):
                ph = blend(ml["Probabilidad_Local"], mc_home)
                pa = blend(ml["Probabilidad_Visita"], mc_away)
                r = evaluar_dos_vias("Moneyline Local", "Moneyline Visitante", ph, pa, market["ml_home"], market["ml_away"], min_prob=55, min_edge=4, min_ev=3)
                diagnostics += consensus_filter(r, ml["Probabilidad_Local"], mc_home, ml["Probabilidad_Visita"], mc_away)

            st.markdown("### Total de carreras")
            st.write({
                "Línea": market.get("total_line"),
                "ML proyectado": ml["Proyeccion_Carreras"],
                "MC promedio": carreras.get("Promedio_Total"),
                "ML Over%": None if ml_over is None else round(ml_over, 1),
                "MC Over%": mc_over,
            })
            if market.get("total_over") and market.get("total_under") and ml_over is not None and ml_under is not None and mc_over is not None and mc_under is not None:
                po = blend(ml_over, mc_over)
                pu = blend(ml_under, mc_under)
                push = carreras.get(f"Push {float(line)}", 0.0)
                r = evaluar_dos_vias(f"Over {line}", f"Under {line}", po, pu, market["total_over"], market["total_under"], push, push, min_prob=57, min_edge=5, min_ev=4)
                diagnostics += consensus_filter(r, ml_over, mc_over, ml_under, mc_under, min_model=55, max_disagreement=12)

            st.markdown("### Run line — diagnóstico")
            if market.get("spread_home") is not None:
                st.write({
                    "Local": market["spread_home"],
                    "MC Local cover%": carreras.get(f"Spread Local {float(market['spread_home']):+.1f}"),
                    "Visitante": market.get("spread_away"),
                    "MC Visitante cover%": carreras.get(f"Spread Visita {float(market['spread_away']):+.1f}") if market.get("spread_away") is not None else None,
                    "Nota": "No se publica pick run line hasta validar el modelo OOS específico de spread.",
                })

            st.markdown("### Diagnóstico de valor")
            if diagnostics:
                diag = pd.DataFrame(diagnostics)
                if proxy and "Estado" in diag.columns:
                    diag.loc[diag.Estado == "VALUE BET", "Estado"] = "SEÑAL — DATOS PROXY"
                st.dataframe(diag, use_container_width=True, hide_index=True)
            else:
                st.info("Sin cuotas completas de una misma casa: no se calcula value bet.")

            with st.expander("Detalle técnico"):
                st.write("Cuotas", market)
                st.write("Monte Carlo", mc)
                st.write("ML", {**ml, "Prob_Over": ml_over, "Prob_Under": ml_under})

        except Exception as exc:
            st.error(f"Error al analizar este partido: {type(exc).__name__}: {exc}")
            with st.expander("Detalle del error"):
                st.code(traceback.format_exc())

st.markdown("---")
st.caption("V2 elimina wRC+/xFIP falsos, H2H arbitrario, clips 35–65%, spreads inventados y API keys dentro del código. El clima direccional permanece neutral hasta incorporar orientación/forecast de estadio verificable.")