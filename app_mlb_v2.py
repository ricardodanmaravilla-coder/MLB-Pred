import datetime
import os
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pandas as pd
import requests
import streamlit as st

from modules.odds_mlb import american_to_decimal, evaluar_dos_vias
from modules.team_utils import normalize_team

st.set_page_config(page_title="MLB Quant Analytics V2", layout="wide", page_icon="⚾")
st.title("⚾ MLB Quant Analytics — Model V2")
st.caption("Navegación liviana · ML se carga solo al analizar · Monte Carlo 10k")

try:
    _secret_key = st.secrets.get("ODDS_API_KEY", "")
except Exception:
    _secret_key = ""
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "") or _secret_key


@st.cache_data(ttl=3600, show_spinner=False)
def load_light_data():
    bat = pd.read_csv("data/mlb_batting.csv")
    pit = pd.read_csv("data/mlb_pitching.csv")
    games = pd.read_csv("data/mlb_games.csv")
    parks = pd.read_csv("data/mlb_park_factors.csv")
    starters = pd.read_csv("data/mlb_pitching_individual.csv") if os.path.exists("data/mlb_pitching_individual.csv") else pd.DataFrame()
    bullpen = pd.read_csv("data/mlb_bullpen.csv") if os.path.exists("data/mlb_bullpen.csv") else pd.DataFrame()
    for df in (bat, pit, parks, starters, bullpen):
        if not df.empty and "Team" in df.columns:
            df["Team"] = df["Team"].map(normalize_team)
    return bat, pit, games, starters, parks, bullpen


@st.cache_resource(show_spinner="Cargando modelo MLB para este servidor...")
def load_trained_model():
    # Important: this function is NEVER called while only navigating the slate.
    from modules.ml_mlb import PredictorMLMLB

    bat = pd.read_csv("data/mlb_batting.csv")
    pit = pd.read_csv("data/mlb_pitching.csv")
    games = pd.read_csv("data/mlb_games.csv")
    model = PredictorMLMLB()
    trained = model.entrenar(bat, pit, games)
    return model, bool(trained)


@st.cache_data(ttl=300, show_spinner=False)
def schedule_today():
    day = datetime.date.today().isoformat()
    r = requests.get(
        "https://statsapi.mlb.com/api/v1/schedule",
        params={"sportId": 1, "date": day, "hydrate": "probablePitcher,team"},
        timeout=12,
    )
    r.raise_for_status()
    out = []
    for block in r.json().get("dates", []):
        for raw in block.get("games", []):
            hobj = raw.get("teams", {}).get("home", {})
            aobj = raw.get("teams", {}).get("away", {})
            hname = hobj.get("team", {}).get("name")
            aname = aobj.get("team", {}).get("name")
            h, a = normalize_team(hname), normalize_team(aname)
            if h and a:
                out.append({
                    "game_id": raw.get("gamePk"),
                    "home_name": hname,
                    "away_name": aname,
                    "home": h,
                    "away": a,
                    "home_pitcher": hobj.get("probablePitcher", {}).get("fullName"),
                    "away_pitcher": aobj.get("probablePitcher", {}).get("fullName"),
                })
    return out


@st.cache_data(ttl=300, show_spinner=False)
def odds_today(api_key):
    if not api_key:
        return []
    r = requests.get(
        "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/",
        params={
            "apiKey": api_key,
            "regions": "us",
            "markets": "h2h,totals,spreads",
            "oddsFormat": "american",
        },
        timeout=12,
    )
    if r.status_code != 200:
        return []
    payload = r.json()
    return payload if isinstance(payload, list) else []


def find_odds(game, feed):
    item = next((x for x in feed if normalize_team(x.get("home_team")) == game["home"] and normalize_team(x.get("away_team")) == game["away"]), None)
    if not item:
        return {}
    for book in item.get("bookmakers", []):
        out = {"bookmaker": book.get("title") or book.get("key") or "N/A"}
        for market in book.get("markets", []):
            key = market.get("key")
            outcomes = market.get("outcomes", [])
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
                    out["total_line"] = float(ov["point"])
                    out["total_over"] = american_to_decimal(ov.get("price"))
                    out["total_under"] = american_to_decimal(un.get("price"))
        if out.get("ml_home") and out.get("ml_away"):
            return out
    return {}


def latest_row(df, team):
    rows = df[df["Team"] == normalize_team(team)].copy()
    if rows.empty:
        return None
    if "Season" in rows.columns:
        rows["Season"] = pd.to_numeric(rows["Season"], errors="coerce")
        rows = rows.sort_values("Season")
    return rows.iloc[-1]


def number(row, *keys):
    if row is None:
        return None
    for key in keys:
        if key in row.index:
            v = pd.to_numeric(row.get(key), errors="coerce")
            if pd.notna(v):
                return float(v)
    return None


def starter_era(starters, name, team, fallback):
    if starters.empty or not name or "Name" not in starters.columns or "ERA" not in starters.columns:
        return fallback, True
    rows = starters[(starters["Team"] == normalize_team(team)) & (starters["Name"].astype(str).str.casefold() == str(name).casefold())]
    if rows.empty:
        return fallback, True
    value = pd.to_numeric(rows.iloc[-1].get("ERA"), errors="coerce")
    return (float(value), False) if pd.notna(value) else (fallback, True)


def bullpen_era(bullpen, team, fallback):
    if bullpen.empty or "Bullpen_ERA" not in bullpen.columns:
        return fallback, True
    rows = bullpen[bullpen["Team"] == normalize_team(team)]
    if rows.empty:
        return fallback, True
    value = pd.to_numeric(rows.iloc[-1].get("Bullpen_ERA"), errors="coerce")
    return (float(value), False) if pd.notna(value) else (fallback, True)


def park_info(parks, team):
    rows = parks[parks["Team"] == normalize_team(team)]
    if rows.empty:
        return 100.0, 0.0
    pf = pd.to_numeric(rows.iloc[-1].get("Park_Factor"), errors="coerce")
    alt = pd.to_numeric(rows.iloc[-1].get("Altitud"), errors="coerce")
    return float(pf) if pd.notna(pf) else 100.0, float(alt) if pd.notna(alt) else 0.0


def blend(p_ml, p_mc):
    return 0.55 * float(p_ml) + 0.45 * float(p_mc)


try:
    bat, pit, games, starters, parks, bullpen = load_light_data()
except Exception as exc:
    st.error(f"Error cargando datos: {type(exc).__name__}: {exc}")
    st.code(traceback.format_exc())
    st.stop()

st.sidebar.metric("Juegos históricos", f"{len(games):,}")
st.sidebar.write(f"ODDS_API_KEY: **{'configurada' if ODDS_API_KEY else 'faltante'}**")
st.sidebar.caption("El ML NO se carga al cambiar de partido")

try:
    slate = schedule_today()
except Exception as exc:
    st.error(f"MLB StatsAPI: {type(exc).__name__}: {exc}")
    slate = []

try:
    market_feed = odds_today(ODDS_API_KEY)
except Exception as exc:
    st.warning(f"The Odds API: {type(exc).__name__}: {exc}")
    market_feed = []

if not slate:
    st.info("No hay juegos MLB programados para hoy.")
    st.stop()

labels = [f"{g['away_name']} @ {g['home_name']}" for g in slate]
choice = st.selectbox("Partido", range(len(slate)), format_func=lambda i: labels[i], key="mlb_game_choice")
game = slate[choice]
market = find_odds(game, market_feed)

st.caption(f"Seleccionado: {labels[choice]} · Abridores: {game.get('away_pitcher') or 'TBA'} / {game.get('home_pitcher') or 'TBA'}")

if st.button("Analizar partido V2", type="primary"):
    try:
        with st.spinner("Cargando ML y analizando partido..."):
            model, trained = load_trained_model()
            if not trained:
                raise RuntimeError("El modelo no pudo entrenar")

            bat_h, bat_a = latest_row(bat, game["home"]), latest_row(bat, game["away"])
            pit_h, pit_a = latest_row(pit, game["home"]), latest_row(pit, game["away"])
            ops_h, ops_a = number(bat_h, "ops"), number(bat_a, "ops")
            era_h, era_a = number(pit_h, "ERA", "era"), number(pit_a, "ERA", "era")
            if None in (ops_h, ops_a, era_h, era_a):
                raise ValueError("Faltan OPS/ERA del equipo")

            sp_h, psh = starter_era(starters, game.get("home_pitcher"), game["home"], era_h)
            sp_a, psa = starter_era(starters, game.get("away_pitcher"), game["away"], era_a)
            bp_h, pbh = bullpen_era(bullpen, game["home"], era_h)
            bp_a, pba = bullpen_era(bullpen, game["away"], era_a)
            proxy = psh or psa or pbh or pba
            pf, altitude = park_info(parks, game["home"])
            line = float(market.get("total_line") or 8.5)

            from modules.montecarlo_mlb import simular_partido_mlb

            mc = simular_partido_mlb(
                game["home"], game["away"],
                ops_loc=ops_h, ops_vis=ops_a,
                pitcher_loc_era=sp_h, pitcher_vis_era=sp_a,
                bullpen_loc_era=bp_h, bullpen_vis_era=bp_a,
                park_factor=pf, altitud_ft=altitude,
                viento_mph=0, direccion_viento="", temp_f=72,
                linea_carreras_casino=line, df_games=games,
                num_simulaciones=10000,
            )
            ml = model.predecir_partido(game["home"], game["away"], ops_h, ops_a, era_h, era_a, pf)

        money = mc.get("Moneyline", {})
        runs = mc.get("Carreras", {})
        mc_home, mc_away = money.get("Gana Local"), money.get("Gana Visita")
        if mc_home is None or mc_away is None:
            raise RuntimeError("Monte Carlo no devolvió moneyline")

        ml_over = model.prob_total(ml["Proyeccion_Carreras"], line, "over")
        ml_under = model.prob_total(ml["Proyeccion_Carreras"], line, "under")
        mc_over = runs.get(f"Over {float(line)}")
        mc_under = runs.get(f"Under {float(line)}")

        st.subheader(labels[choice])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("OPS local", f"{ops_h:.3f}")
        c2.metric("OPS visita", f"{ops_a:.3f}")
        c3.metric("ERA abridor local", f"{sp_h:.2f}")
        c4.metric("ERA abridor visita", f"{sp_a:.2f}")
        if proxy:
            st.warning("Hay datos proxy de abridor/bullpen; trate cualquier value bet como señal.")

        ph = blend(ml["Probabilidad_Local"], mc_home)
        pa = blend(ml["Probabilidad_Visita"], mc_away)
        st.markdown("### Moneyline")
        st.dataframe(pd.DataFrame([
            {"Lado": "Local", "ML %": ml["Probabilidad_Local"], "MC %": mc_home, "Ensemble %": round(ph, 1)},
            {"Lado": "Visitante", "ML %": ml["Probabilidad_Visita"], "MC %": mc_away, "Ensemble %": round(pa, 1)},
        ]), use_container_width=True, hide_index=True)

        diagnostics = []
        if market.get("ml_home") and market.get("ml_away"):
            diagnostics += evaluar_dos_vias("Moneyline Local", "Moneyline Visitante", ph, pa, market["ml_home"], market["ml_away"], min_prob=55, min_edge=4, min_ev=3)

        st.markdown("### Total de carreras")
        st.write({
            "Línea": market.get("total_line") or "Sin línea real (diagnóstico 8.5)",
            "ML proyectado": ml["Proyeccion_Carreras"],
            "MC promedio": runs.get("Promedio_Total"),
            "ML Over %": None if ml_over is None else round(ml_over, 1),
            "MC Over %": mc_over,
        })
        if market.get("total_over") and market.get("total_under") and None not in (ml_over, ml_under, mc_over, mc_under):
            push = runs.get(f"Push {float(line)}", 0.0)
            diagnostics += evaluar_dos_vias(f"Over {line}", f"Under {line}", blend(ml_over, mc_over), blend(ml_under, mc_under), market["total_over"], market["total_under"], push, push, min_prob=57, min_edge=5, min_ev=4)

        st.markdown("### Diagnóstico de valor")
        if diagnostics:
            diag = pd.DataFrame(diagnostics)
            if proxy and "Estado" in diag.columns:
                diag.loc[diag["Estado"] == "VALUE BET", "Estado"] = "SEÑAL — DATOS PROXY"
            st.dataframe(diag, use_container_width=True, hide_index=True)
        else:
            st.info("Sin cuotas completas de una misma casa: se muestran probabilidades, pero no value bet.")

        with st.expander("Detalle técnico"):
            st.write("Cuotas", market)
            st.write("Monte Carlo", mc)
            st.write("ML", ml)

    except Exception as exc:
        st.error(f"Error al analizar: {type(exc).__name__}: {exc}")
        with st.expander("Detalle del error"):
            st.code(traceback.format_exc())

st.markdown("---")
st.caption("V2 lazy: cambiar de partido no carga ni entrena ML. El ML se solicita únicamente al pulsar Analizar.")
