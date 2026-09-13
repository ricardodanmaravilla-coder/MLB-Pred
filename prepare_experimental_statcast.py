"""Build real Baseball Savant / Statcast team batting features for research only.

This script is intentionally isolated from production. It downloads pitch-level
Statcast data with pybaseball, reduces it immediately to daily team aggregates,
and enriches the experimental Parquet with 30-calendar-day pregame features.
Only dates STRICTLY BEFORE each target game are used, so same-day and future
information can never leak into the historical backtest.

Real Statcast fields used:
- estimated_woba_using_speedangle + woba_value/woba_denom -> expected wOBA-like PA rate
- estimated_slg_using_speedangle -> expected SLG on contacted balls
- launch_speed >= 95 mph -> HardHit%
- launch_speed_angle == 6 -> Barrel%
- launch_speed -> average exit velocity

No value is fabricated. Missing source coverage stays NaN.
"""
from __future__ import annotations

from pathlib import Path
import json
import random
import time

import numpy as np
import pandas as pd
from pybaseball import statcast

from modules.experimental_parquet import OUT, REPORT
from modules.team_utils import normalize_team

DAILY_OUT = Path("data/experimental/statcast_team_daily_real.csv")
START_DATE = pd.Timestamp("2021-03-15")
CHUNK_DAYS = 35
MIN_XWOBA_PA = 25
MIN_BBE = 15
FETCH_RETRIES = 6
FETCH_BACKOFF_BASE_SECONDS = 4.0
FAILED_FETCHES: list[dict[str, str]] = []


def _num(s):
    """Return a plain float64 Series, never pandas nullable integer dtype."""
    if isinstance(s, pd.Series):
        return pd.Series(pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64", na_value=np.nan), index=s.index)
    return pd.Series(pd.to_numeric(pd.Series(s), errors="coerce").to_numpy(dtype="float64", na_value=np.nan))


def _fetch_chunk_once(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch one date range with retry/backoff and a serial fallback.

    Baseball Savant occasionally answers with 5xx/502 errors. We retry with
    exponential backoff plus jitter. Later attempts disable pybaseball's
    parallel mode because smaller/serial requests are often more reliable when
    Savant is under load.
    """
    err = None
    for attempt in range(FETCH_RETRIES):
        parallel = attempt < 3
        try:
            print(
                f"Statcast {start.date()} -> {end.date()} "
                f"intento={attempt + 1}/{FETCH_RETRIES} parallel={parallel}"
            )
            df = statcast(
                start_dt=start.date().isoformat(),
                end_dt=end.date().isoformat(),
                verbose=False,
                parallel=parallel,
            )
            return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        except Exception as exc:
            err = exc
            if attempt + 1 < FETCH_RETRIES:
                delay = FETCH_BACKOFF_BASE_SECONDS * (2 ** attempt) + random.uniform(0.0, 2.0)
                delay = min(delay, 75.0)
                print(
                    f"WARN Statcast temporal {start.date()}..{end.date()}: {exc}. "
                    f"Reintento en {delay:.1f}s"
                )
                time.sleep(delay)
    raise RuntimeError(f"Statcast falló {start.date()}..{end.date()}: {err}")


def _fetch_chunk(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch robustly, recursively reducing a failed range.

    A persistent failure for a large chunk is split in half. If even a single
    day remains unavailable after all retries, that day is recorded and skipped
    rather than fabricating data or aborting the entire Shadow research build.
    The resulting missing historical coverage therefore remains NaN downstream.
    """
    try:
        return _fetch_chunk_once(start, end)
    except Exception as exc:
        if start < end:
            span_days = int((end - start).days)
            mid = start + pd.Timedelta(days=span_days // 2)
            print(
                f"WARN Statcast rango persistente {start.date()}..{end.date()}; "
                f"dividiendo en {start.date()}..{mid.date()} y "
                f"{(mid + pd.Timedelta(days=1)).date()}..{end.date()}"
            )
            left = _fetch_chunk(start, mid)
            right = _fetch_chunk(mid + pd.Timedelta(days=1), end)
            if left.empty:
                return right
            if right.empty:
                return left
            return pd.concat([left, right], ignore_index=True)

        failure = {
            "start": str(start.date()),
            "end": str(end.date()),
            "error": str(exc),
        }
        FAILED_FETCHES.append(failure)
        print(
            f"WARN Statcast indisponible para {start.date()} tras todos los reintentos; "
            "se conserva como cobertura faltante, sin datos inventados."
        )
        return pd.DataFrame()


def _daily_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    need = {"game_date", "inning_topbot", "home_team", "away_team"}
    if not need.issubset(raw.columns):
        raise RuntimeError(f"Statcast schema incompleto: faltan {sorted(need-set(raw.columns))}")

    x = raw.copy()
    x["Date"] = pd.to_datetime(x["game_date"], errors="coerce").dt.normalize()
    top = x["inning_topbot"].astype(str).str.casefold().str.startswith("top")
    x["Team"] = np.where(top, x["away_team"], x["home_team"])
    x["Team"] = pd.Series(x["Team"], index=x.index).map(normalize_team)

    events = x["events"] if "events" in x.columns else pd.Series(np.nan, index=x.index)
    pa = x[events.notna()].copy()
    if not pa.empty:
        denom = _num(pa["woba_denom"]) if "woba_denom" in pa.columns else pd.Series(np.nan, index=pa.index, dtype="float64")
        actual = _num(pa["woba_value"]) if "woba_value" in pa.columns else pd.Series(np.nan, index=pa.index, dtype="float64")
        est = _num(pa["estimated_woba_using_speedangle"]) if "estimated_woba_using_speedangle" in pa.columns else pd.Series(np.nan, index=pa.index, dtype="float64")
        pa["_woba_den"] = denom.where(denom > 0, 0.0).fillna(0.0).astype("float64")
        component = est.combine_first(actual).astype("float64")
        pa["_xwoba_num"] = component.fillna(0.0) * pa["_woba_den"]
        pa_daily = pa.groupby(["Date", "Team"], as_index=False).agg(
            xwoba_num=("_xwoba_num", "sum"), woba_den=("_woba_den", "sum")
        )
    else:
        pa_daily = pd.DataFrame(columns=["Date", "Team", "xwoba_num", "woba_den"])

    ev = _num(x["launch_speed"]) if "launch_speed" in x.columns else pd.Series(np.nan, index=x.index, dtype="float64")
    bbe = x[ev.notna()].copy()
    if not bbe.empty:
        bbe["_ev"] = _num(bbe["launch_speed"])
        bbe["_hard"] = (bbe["_ev"] >= 95.0).astype(float)
        if "launch_speed_angle" in bbe.columns:
            lsa = _num(bbe["launch_speed_angle"])
            bbe["_barrel"] = (lsa == 6).astype(float)
        else:
            bbe["_barrel"] = np.nan
        bbe["_xslg"] = _num(bbe["estimated_slg_using_speedangle"]) if "estimated_slg_using_speedangle" in bbe.columns else np.nan
        bbe["_one"] = 1.0
        bbe["_xslg_n"] = bbe["_xslg"].notna().astype(float)
        bbe["_xslg_sum"] = bbe["_xslg"].fillna(0.0)
        bbe_daily = bbe.groupby(["Date", "Team"], as_index=False).agg(
            bbe=("_one", "sum"), ev_sum=("_ev", "sum"),
            hardhit_n=("_hard", "sum"), barrel_n=("_barrel", "sum"),
            xslg_sum=("_xslg_sum", "sum"), xslg_n=("_xslg_n", "sum")
        )
    else:
        bbe_daily = pd.DataFrame(columns=["Date", "Team", "bbe", "ev_sum", "hardhit_n", "barrel_n", "xslg_sum", "xslg_n"])

    d = pa_daily.merge(bbe_daily, on=["Date", "Team"], how="outer")
    return d.dropna(subset=["Date", "Team"])


def build_daily(max_date: pd.Timestamp) -> pd.DataFrame:
    FAILED_FETCHES.clear()
    parts = []
    start = START_DATE
    max_date = pd.Timestamp(max_date).normalize()
    while start <= max_date:
        end = min(start + pd.Timedelta(days=CHUNK_DAYS - 1), max_date)
        raw = _fetch_chunk(start, end)
        d = _daily_from_raw(raw)
        if not d.empty:
            parts.append(d)
        del raw
        start = end + pd.Timedelta(days=1)
    if not parts:
        raise RuntimeError("Statcast no devolvió datos reales")
    daily = pd.concat(parts, ignore_index=True)
    numcols = ["xwoba_num", "woba_den", "bbe", "ev_sum", "hardhit_n", "barrel_n", "xslg_sum", "xslg_n"]
    for c in numcols:
        if c not in daily.columns:
            daily[c] = 0.0
        daily[c] = _num(daily[c]).fillna(0.0)
    daily = daily.groupby(["Date", "Team"], as_index=False)[numcols].sum()
    DAILY_OUT.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(DAILY_OUT, index=False)
    print(f"OK real Statcast daily rows={len(daily)} teams={daily.Team.nunique()} dates={daily.Date.min().date()}..{daily.Date.max().date()}")
    if FAILED_FETCHES:
        print(f"WARN Statcast días/rangos omitidos por indisponibilidad persistente: {len(FAILED_FETCHES)}")
    return daily


def _rolling_map(daily: pd.DataFrame):
    maps = {}
    for team, g in daily.groupby("Team"):
        g = g.sort_values("Date").set_index("Date")
        maps[normalize_team(team)] = g
    return maps


def _features_for(team_map: pd.DataFrame | None, game_date: pd.Timestamp):
    if team_map is None or team_map.empty or pd.isna(game_date):
        return (np.nan,) * 5
    lo = pd.Timestamp(game_date).normalize() - pd.Timedelta(days=30)
    hi = pd.Timestamp(game_date).normalize() - pd.Timedelta(days=1)
    w = team_map.loc[(team_map.index >= lo) & (team_map.index <= hi)]
    if w.empty:
        return (np.nan,) * 5
    wden = float(w["woba_den"].sum())
    bbe = float(w["bbe"].sum())
    xslg_n = float(w["xslg_n"].sum())
    xwoba = float(w["xwoba_num"].sum() / wden) if wden >= MIN_XWOBA_PA else np.nan
    xslg = float(w["xslg_sum"].sum() / xslg_n) if xslg_n >= MIN_BBE else np.nan
    hard = float(w["hardhit_n"].sum() / bbe) if bbe >= MIN_BBE else np.nan
    barrel = float(w["barrel_n"].sum() / bbe) if bbe >= MIN_BBE else np.nan
    ev = float(w["ev_sum"].sum() / bbe) if bbe >= MIN_BBE else np.nan
    return xwoba, xslg, hard, barrel, ev


def enrich_parquet(daily: pd.DataFrame) -> dict:
    if not OUT.exists():
        raise RuntimeError(f"Research Parquet missing: {OUT}")
    frame = pd.read_parquet(OUT)
    dates = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
    maps = _rolling_map(daily)
    metric_names = ["xwoba_30d", "xslg_30d", "hardhit_30d", "barrel_30d", "ev_30d"]
    for side, team_col in (("home", "Home"), ("away", "Away")):
        vals = []
        for team, d in zip(frame[team_col], dates):
            vals.append(_features_for(maps.get(normalize_team(team)), d))
        for j, metric in enumerate(metric_names):
            frame[f"{side}_{metric}"] = [v[j] for v in vals]
    frame.to_parquet(OUT, index=False)

    coverage = {f"{side}_{m}": round(float(frame[f"{side}_{m}"].notna().mean()), 4)
                for side in ("home", "away") for m in metric_names}
    payload = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {}
    payload.setdefault("coverage_after_context", {}).update(coverage)
    payload["statcast_source"] = {
        "provider": "Baseball Savant via pybaseball.statcast",
        "source_type": "real pitch-level Statcast",
        "daily_file": str(DAILY_OUT),
        "window": "30 calendar days strictly before game date",
        "start_date": str(START_DATE.date()),
        "max_date": str(pd.to_datetime(frame["Date"]).max().date()),
        "metrics": metric_names,
        "xwoba_note": "expected contact wOBA for BIP plus deterministic wOBA event values for non-contact PA",
        "xslg_note": "mean Statcast estimated SLG on contacted balls",
        "leakage_guard": "target game date and future dates excluded",
        "fetch_retries": FETCH_RETRIES,
        "persistent_fetch_failures": FAILED_FETCHES,
        "persistent_fetch_failure_count": len(FAILED_FETCHES),
        "missing_data_policy": "persistent unavailable source dates remain missing; no synthetic values",
    }
    REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"rows": len(frame), "coverage": coverage, "persistent_fetch_failures": len(FAILED_FETCHES)}, indent=2))
    return coverage


def main():
    if not OUT.exists():
        raise RuntimeError("Build experimental Parquet before Statcast enrichment")
    frame = pd.read_parquet(OUT, columns=["Date"])
    max_date = pd.to_datetime(frame["Date"], errors="coerce").max()
    if pd.isna(max_date):
        raise RuntimeError("Parquet has no valid dates")
    daily = build_daily(max_date)
    enrich_parquet(daily)


if __name__ == "__main__":
    main()
