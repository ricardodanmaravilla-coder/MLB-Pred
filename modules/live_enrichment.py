"""Fail-soft live MLB context enrichment.

These signals are intentionally separated from the validated baseline model.  They can
run in shadow mode so we can measure them before allowing them to change a production
pick.  No missing advanced statistic is fabricated; unavailable inputs stay neutral.
"""
from __future__ import annotations

import math
import unicodedata
from typing import Any

import numpy as np
import pandas as pd

from .metric_quality import row_pitching_value
from .team_utils import normalize_team

LOWER_IS_BETTER = {
    "FIP": 0.18,
    "xFIP": 0.28,
    "SIERA": 0.20,
    "WHIP": 0.12,
    "BB%": 0.05,
    "HR/9": 0.07,
}
HIGHER_IS_BETTER = {
    "K-BB%": 0.07,
    "K%": 0.03,
}


def _num(value: Any):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _name_key(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.casefold().replace(".", " ").replace("-", " ").split())


def starter_row(pitchers: pd.DataFrame, name: str, team: str | None = None):
    """Resolve a probable starter without silently accepting ambiguous surnames."""
    if pitchers is None or pitchers.empty or not name or name == "Por Anunciar" or "Name" not in pitchers.columns:
        return None
    x = pitchers.copy()
    x["_name_key"] = x["Name"].map(_name_key)
    key = _name_key(name)
    match = x[x["_name_key"] == key]
    if team and "Team" in x.columns:
        team_key = normalize_team(team)
        exact_team = match[match["Team"].map(normalize_team) == team_key]
        if not exact_team.empty:
            match = exact_team
    if match.empty:
        surname = key.split()[-1] if key else ""
        fallback = x[x["_name_key"].str.split().str[-1] == surname]
        if team and "Team" in fallback.columns:
            team_key = normalize_team(team)
            team_rows = fallback[fallback["Team"].map(normalize_team) == team_key]
            if not team_rows.empty:
                fallback = team_rows
        if fallback["Name"].nunique() != 1:
            return None
        match = fallback
    if "Season" in match.columns:
        match = match.assign(_season=pd.to_numeric(match["Season"], errors="coerce")).sort_values("_season")
    return match.iloc[-1]


def starter_profile(pitchers: pd.DataFrame, name: str, team: str | None = None) -> dict:
    """Build a conservative multi-metric starter profile.

    The profile expresses advanced metrics as relative factors versus the same-season
    pitcher population.  It is suitable for shadow testing; the production baseline
    remains unchanged unless an explicit validated-enrichment mode is enabled.
    """
    row = starter_row(pitchers, name, team)
    if row is None:
        return {
            "resolved": False, "name": name, "hand": None, "base_metric": None,
            "composite_metric": None, "coverage": 0.0, "metrics_used": [],
        }
    base, base_source = row_pitching_value(row, None)
    hand = str(row.get("PitchHand", "")).strip().upper()[:1]
    if hand not in {"L", "R"}:
        hand = None
    population = pitchers.copy()
    if "Season" in population.columns and pd.notna(row.get("Season")):
        season = pd.to_numeric(pd.Series([row.get("Season")]), errors="coerce").iloc[0]
        if pd.notna(season):
            pseason = pd.to_numeric(population["Season"], errors="coerce")
            same = population[pseason == float(season)]
            if len(same) >= 30:
                population = same
    weighted_log = 0.0
    weight_sum = 0.0
    used = []
    for col, weight in {**LOWER_IS_BETTER, **HIGHER_IS_BETTER}.items():
        if col not in population.columns:
            continue
        value = _num(row.get(col))
        series = pd.to_numeric(population[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if value is None or len(series) < 30:
            continue
        center = float(series.median())
        if abs(center) < 1e-9:
            continue
        if col in LOWER_IS_BETTER:
            factor = value / center
        else:
            # Higher strikeout-quality metrics imply better run prevention, so invert.
            factor = center / value if abs(value) > 1e-9 else 1.0
        factor = float(np.clip(factor, 0.82, 1.18))
        weighted_log += float(weight) * math.log(factor)
        weight_sum += float(weight)
        used.append(col)
    adjustment = math.exp(weighted_log / weight_sum) if weight_sum > 0 else 1.0
    composite = None if base is None else float(np.clip(float(base) * adjustment, 1.75, 7.50))
    total_possible = sum(LOWER_IS_BETTER.values()) + sum(HIGHER_IS_BETTER.values())
    coverage = 0.0 if total_possible <= 0 else min(1.0, weight_sum / total_possible)
    return {
        "resolved": True,
        "name": str(row.get("Name", name)),
        "hand": hand,
        "base_metric": None if base is None else round(float(base), 4),
        "base_source": base_source,
        "composite_metric": None if composite is None else round(composite, 4),
        "adjustment_factor": round(float(adjustment), 4),
        "coverage": round(float(coverage), 4),
        "metrics_used": used,
    }


def platoon_offense_index(batting: pd.DataFrame, team: str, pitcher_hand: str | None, fallback: float = 100.0) -> dict:
    """Return a team offensive index against the actual starter hand when available."""
    hand = str(pitcher_hand or "").strip().upper()[:1]
    if batting is None or batting.empty or hand not in {"L", "R"} or "Team" not in batting.columns:
        return {"index": float(fallback), "hand": hand or None, "used": False, "coverage": 0.0}
    x = batting.copy()
    x["_team"] = x["Team"].map(normalize_team)
    if "Season" in x.columns:
        x["_season"] = pd.to_numeric(x["Season"], errors="coerce")
        latest = x["_season"].dropna().max()
        if pd.notna(latest):
            season = x[x["_season"] == latest].copy()
        else:
            season = x
    else:
        season = x
    rows = season[season["_team"] == normalize_team(team)]
    if rows.empty:
        return {"index": float(fallback), "hand": hand, "used": False, "coverage": 0.0}
    row = rows.iloc[-1]
    cols = [(f"OPS_vs_{hand}", 0.50), (f"OBP_vs_{hand}", 0.25), (f"SLG_vs_{hand}", 0.25)]
    logs = 0.0
    weight_sum = 0.0
    used = []
    for col, weight in cols:
        if col not in season.columns:
            continue
        val = _num(row.get(col))
        series = pd.to_numeric(season[col], errors="coerce").dropna()
        if val is None or len(series) < 20:
            continue
        center = float(series.median())
        if center <= 0:
            continue
        ratio = float(np.clip(val / center, 0.80, 1.20))
        logs += weight * math.log(ratio)
        weight_sum += weight
        used.append(col)
    if weight_sum <= 0:
        return {"index": float(fallback), "hand": hand, "used": False, "coverage": 0.0}
    relative = math.exp(logs / weight_sum)
    # Blend with the validated team baseline instead of replacing it outright.
    enriched = float(np.clip(float(fallback) * (0.65 + 0.35 * relative), 75.0, 125.0))
    return {
        "index": round(enriched, 4), "hand": hand, "used": True,
        "coverage": round(min(1.0, weight_sum), 4), "relative_factor": round(relative, 4),
        "metrics_used": used,
    }


def build_live_enrichment(batting: pd.DataFrame, pitchers: pd.DataFrame, *, home: str, away: str,
                          home_pitcher: str, away_pitcher: str, base_off_home: float,
                          base_off_away: float, base_pitch_home: float, base_pitch_away: float) -> dict:
    home_sp = starter_profile(pitchers, home_pitcher, home)
    away_sp = starter_profile(pitchers, away_pitcher, away)
    # Home offense faces away starter; away offense faces home starter.
    home_bat = platoon_offense_index(batting, home, away_sp.get("hand"), base_off_home)
    away_bat = platoon_offense_index(batting, away, home_sp.get("hand"), base_off_away)
    return {
        "home_starter": home_sp,
        "away_starter": away_sp,
        "home_offense_vs_hand": home_bat,
        "away_offense_vs_hand": away_bat,
        "candidate_inputs": {
            "off_home": float(home_bat.get("index", base_off_home)),
            "off_away": float(away_bat.get("index", base_off_away)),
            "pitch_home": float(home_sp.get("composite_metric") or base_pitch_home),
            "pitch_away": float(away_sp.get("composite_metric") or base_pitch_away),
        },
    }
