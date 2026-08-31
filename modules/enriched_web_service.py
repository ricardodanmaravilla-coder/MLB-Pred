"""Validated/live enrichment wrapper for the Cloud Run service.

The existing MLBWebService remains the immutable production baseline.  This wrapper
collects platoon and multi-metric starter information in SHADOW mode by default.
Only MLB_ENRICHMENT_MODE=validated allows those candidate inputs to alter Monte Carlo.
That separation prevents an untested statistic from silently changing picks.
"""
from __future__ import annotations

import os
import threading

from .live_enrichment import build_live_enrichment
from .web_service import EQUIPOS_MAP, MLBWebService


class EnrichedMLBWebService(MLBWebService):
    def __init__(self):
        self._enrichment_local = threading.local()
        super().__init__()

    @property
    def enrichment_mode(self) -> str:
        mode = str(os.getenv("MLB_ENRICHMENT_MODE", "shadow")).strip().lower()
        return "validated" if mode in {"validated", "on", "enabled", "1", "true"} else "shadow"

    def health(self):
        data = super().health()
        data["enrichment_mode"] = self.enrichment_mode
        data["enrichment_policy"] = "validated_only"
        return data

    def _active_enrichment(self):
        return getattr(self._enrichment_local, "value", None)

    def _current_offensive_index(self, team, fallback=100.0):
        baseline = super()._current_offensive_index(team, fallback)
        if self.enrichment_mode != "validated":
            return baseline
        ctx = self._active_enrichment() or {}
        inputs = ctx.get("candidate_inputs", {})
        home = ctx.get("home")
        away = ctx.get("away")
        if team == home:
            return float(inputs.get("off_home", baseline))
        if team == away:
            return float(inputs.get("off_away", baseline))
        return baseline

    def _starter_metric(self, name):
        baseline = super()._starter_metric(name)
        if self.enrichment_mode != "validated":
            return baseline
        ctx = self._active_enrichment() or {}
        if str(name) == str(ctx.get("home_pitcher")):
            return float(ctx.get("candidate_inputs", {}).get("pitch_home") or baseline) if baseline is not None else ctx.get("candidate_inputs", {}).get("pitch_home")
        if str(name) == str(ctx.get("away_pitcher")):
            return float(ctx.get("candidate_inputs", {}).get("pitch_away") or baseline) if baseline is not None else ctx.get("candidate_inputs", {}).get("pitch_away")
        return baseline

    def _prepare_enrichment(self, game):
        home_name, away_name = game.get("home"), game.get("away")
        home, away = EQUIPOS_MAP.get(home_name, ""), EQUIPOS_MAP.get(away_name, "")
        base_off_home = super()._current_offensive_index(home) if home else 100.0
        base_off_away = super()._current_offensive_index(away) if away else 100.0
        base_pitch_home = super()._starter_metric(game.get("home_pitcher")) if home else None
        base_pitch_away = super()._starter_metric(game.get("away_pitcher")) if away else None
        if base_pitch_home is None and home:
            base_pitch_home = super()._team_pitching(home)
        if base_pitch_away is None and away:
            base_pitch_away = super()._team_pitching(away)
        enrichment = build_live_enrichment(
            self.batting, self.pitchers,
            home=home, away=away,
            home_pitcher=game.get("home_pitcher"), away_pitcher=game.get("away_pitcher"),
            base_off_home=base_off_home, base_off_away=base_off_away,
            base_pitch_home=base_pitch_home or 4.10, base_pitch_away=base_pitch_away or 4.10,
        )
        enrichment.update({
            "home": home, "away": away,
            "home_pitcher": game.get("home_pitcher"),
            "away_pitcher": game.get("away_pitcher"),
            "mode": self.enrichment_mode,
            "affects_pick": self.enrichment_mode == "validated",
        })
        return enrichment

    def _evaluate_game(self, game):
        enrichment = self._prepare_enrichment(game)
        self._enrichment_local.value = enrichment
        try:
            result = super()._evaluate_game(game)
        finally:
            self._enrichment_local.value = None
        result.setdefault("context", {})["enrichment"] = enrichment
        # Put auditable signal metadata on each candidate without changing the
        # scanner's mathematical thresholds or market/EV/Kelly calculations.
        for row in result.get("diagnostics", []):
            row["enrichment_mode"] = self.enrichment_mode
            row["starter_hand_home"] = enrichment.get("home_starter", {}).get("hand")
            row["starter_hand_away"] = enrichment.get("away_starter", {}).get("hand")
            row["starter_profile_coverage_home"] = enrichment.get("home_starter", {}).get("coverage", 0.0)
            row["starter_profile_coverage_away"] = enrichment.get("away_starter", {}).get("coverage", 0.0)
            row["platoon_used_home"] = bool(enrichment.get("home_offense_vs_hand", {}).get("used"))
            row["platoon_used_away"] = bool(enrichment.get("away_offense_vs_hand", {}).get("used"))
        for row in result.get("accepted", []):
            row["enrichment_mode"] = self.enrichment_mode
        return result
