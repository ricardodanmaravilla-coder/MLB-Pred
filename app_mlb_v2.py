import datetime
import os
import traceback

# Limit native numerical threads before importing sklearn/numpy through modules.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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
    _secret_key = st.secrets.get("ODDS_API_KEY", "")
except Exception:
    _secret_key = ""
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "") or _secret_key


@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
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


@st.cache_resource(show_spinner="Entrenando modelo MLB V2 una sola vez...")
def load_trained_model():
    # Train inside cache_resource so Streamlit reruns do NOT retrain the model.
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
            home_obj = raw.get("teams", {}).get("home", {})
            away_obj = raw.get("teams", {}).get("away", {})
            home_name = home_obj.get("team", {}).get("name")
            away_name = away_obj.get("team", {}).get("name")
            home = normalize_team(home_name)
            away = normalize_team(away_name)
            if home and away:
                out.append({
                    "game_id": raw.get("gamePk"),
                    "home_name": home_name,
                    "away_name": away_name,
                    "home": home,
                    "away": away,
                    "home_pitcher": home_obj.get("probablePitcher", {}).get("fullName"),
                    "away_pitcher": away_obj.get("probablePitcher", {}).get("fullName"),
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
    item = next(
        (
            x for x in feed
            if normalize_team(x.get("home_team")) == game["home"]
            and normalize_team(x.get("away_team")) == game["away"]
        ),
        None,
    )
    if not item:
        return {}
    for book in item.get("bookmakers", []):
        out = {"bookmaker": book.get("title") or book.get("key") or "N/A"}
        for market in book.get("markets", []):
            key = market.get("key")
            outcomes = market.get("outcomes", [])
            if key == "h2h":
                for o in outcomes:
                    team = normalize_team(o.get("name"))
                    if team == game["home"]:
                        out["ml_home"] = american_to_decimal(o.get("price"))
                    elif team == game["away"]:
                        out["ml_away"] = american_to_decimal(o.get("price"))
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
                        out["spread_home"] = float(o.get("point"))
                    elif team == game["away"]:
                        out["spread_away"] = float(o.get("point"))
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
            value = pd.to_numeric(row.get(key), errors="coerce")
            if pd.notna(value):
                return float(value)
    return None


def starter_era(starters, name, team, fallback):
    if starters.empty or not name or "Name" not in starters.columns or "ERA" not in starters.columns:
        return fallback, True
    rows = starters[
        (starters["Team"] == normalize_team(team))
        & (starters["Name"].astype(str).str.casefold() == str(name).casefold())
    ]
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
    return (
        float(pf) if pd.notna(pf) else 100.0,
        float(alt) if pd.notna(alt) else 0.0,
    )


def blend(p_ml, p_mc):
    return 0.55 * float(p_ml) + 0.45 * float(p_mc)


try:
    bat, pit, games, starters, parks, bullpen = load_data()
    model, trained = load_trained_model()
except Exception as exc:
    st.error(f"Error inicializando MLB V2: {type(exc).__name__}: {exc}")
    st.code(traceback.format_exc())
    st.stop()

st.sidebar.metric("Juegos históricos", f"{len(games):,}")
st.sidebar.write(f"ML calibrado: **{'Sí' if trained and model.calibrado else 'No'}**")
st.sidebar.write(f"ODDS_API_KEY: **{'configurada' if ODDS_API_KEY else 'faltante'}**")
st.sidebar.caption("Modelo cacheado entre clics · Monte Carlo 20k por análisis")

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
choice = st.selectbox("Partido", range(len(slate)), format_func=lambda i: labels[i])
game = slate[choice]
market = find_odds(game, market_feed)

if st.button("Analizar partido V2", type="primary"):
    try:
        if not trained:
            raise RuntimeError("El modelo no pudo entrenar")

        bat_h, bat_a = latest_row(bat, game["home"]), latest_row(bat, game["away"])
        pit_h, pit_a = latest_row(pit, game["home"]), latest_row(pit, game["away"])
        ops_h, ops_a = number(bat_h, "ops"), number(bat_a, "ops")
        era_h, era_a = number(pit_h, "ERA", "era"), number(pit_a, "ERA", "era")
        if None in (ops_h, ops_a, era_h, era_a):
            raise ValueError("Faltan OPS/ERA del equipo")

        sp_h, proxy_sp_h = starter_era(starters, game.get("home_pitcher"), game["home"], era_h)
        sp_a, proxy_sp_a = starter_era(starters, game.get("away_pitcher"), game["away"], era_a)
        bp_h, proxy_bp_h = bullpen_era(bullpen, game["home"], era_h)
        bp_a, proxy_bp_a = bullpen_era(bullpen, game["away"], era_a)
        pf, altitude = park_info(parks, game["home"])
        proxy = proxy_sp_h or proxy_sp_a or proxy_bp_h or proxy_bp_a
        line = float(market.get("total_line") or 8.5)

        with st.spinner("Analizando partido..."):
            mc = simular_partido_mlb(
                game["home"],
                game["away"],
                ops_loc=ops_h,
                ops_vis=ops_a,
                pitcher_loc_era=sp_h,
                pitcher_vis_era=sp_a,
                bullpen_loc_era=bp_h,
                bullpen_vis_era=bp_a,
                park_factor=pf,
                altitud_ft=altitude,
                viento_mph=0,
                direccion_viento="",
                temp_f=72,
                linea_carreras_casino=line,
                df_games=games,
                spread_loc=market.get("spread_home"),
                spread_vis=market.get("spread_away"),
                num_simulaciones=20000,
            )
            ml = model.predecir_partido(
                game["home"], game["away"], ops_h, ops_a, era_h, era_a, pf
            )

        money = mc.get("Moneyline", {})
        runs = mc.get("Carreras", {})
        mc_home = money.get("Gana Local")
        mc_away = money.get("Gana Visita")
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
        st.caption(
            f"Abridores: {game.get('home_pitcher') or 'TBA'} / {game.get('away_pitcher') or 'TBA'} · "
            f"Park factor {pf:.0f} · Bookmaker {market.get('bookmaker', 'N/A')}"
        )
        if proxy:
            st.warning("Hay datos proxy de abridor/bullpen; cualquier value bet debe tratarse como señal, no como pick fuerte.")

        st.markdown("### Moneyline")
        ph = blend(ml["Probabilidad_Local"], mc_home)
        pa = blend(ml["Probabilidad_Visita"], mc_away)
        st.dataframe(
            pd.DataFrame([
                {"Lado": "Local", "ML %": ml["Probabilidad_Local"], "MC %": mc_home, "Ensemble %": round(ph, 1)},
                {"Lado": "Visitante", "ML %": ml["Probabilidad_Visita"], "MC %": mc_away, "Ensemble %": round(pa, 1)},
            ]),
            use_container_width=True,
            hide_index=True,
        )

        diagnostics = []
        if market.get("ml_home") and market.get("ml_away"):
            diagnostics += evaluar_dos_vias(
                "Moneyline Local",
                "Moneyline Visitante",
                ph,
                pa,
                market["ml_home"],
                market["ml_away"],
                min_prob=55,
                min_edge=4,
                min_ev=3,
            )

        st.markdown("### Total de carreras")
        st.write({
            "Línea": market.get("total_line") or "Sin línea real (diagnóstico 8.5)",
            "ML proyectado": ml["Proyeccion_Carreras"],
            "MC promedio": runs.get("Promedio_Total"),
            "ML Over %": None if ml_over is None else round(ml_over, 1),
            "MC Over %": mc_over,
        })
        if (
            market.get("total_over")
            and market.get("total_under")
            and ml_over is not None
            and ml_under is not None
            and mc_over is not None
            and mc_under is not None
        ):
            diagnostics += evaluar_dos_vias(
                f"Over {line}",
                f"Under {line}",
                blend(ml_over, mc_over),
                blend(ml_under, mc_under),
                market["total_over"],
                market["total_under"],
                runs.get(f"Push {float(line)}", 0.0),
                runs.get(f"Push {float(line)}", 0.0),
                min_prob=57,
                min_edge=5,
                min_ev=4,
            )

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
st.caption("V2 estable: modelo cacheado, Monte Carlo 20k y errores visibles. No se inventan cuotas ni líneas para publicar value bets.")