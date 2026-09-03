"""Validated/live enrichment wrapper for the Cloud Run service.

V7 remains the production baseline. The walk-forward winner
starter_day_night + team_context runs in a separate shadow path and writes only
to MLB_Candidate_Picks. It never changes V7 recommendations.
"""
from __future__ import annotations

import os
import threading

import pandas as pd

from .live_enrichment import build_live_enrichment
from .pick_ledger import append_snapshot, append_shadow_snapshot
from .scanner_engine import moneyline_candidate, no_vig_two_way, runline_candidate, total_candidate
from .shadow_candidate import available as shadow_available, metadata as shadow_metadata, predict as shadow_predict
from .web_service import EQUIPOS_MAP, MLBWebService, estimate_ml_probability, kelly_fraction_pct
from .game_context import slate_date


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
        data["shadow_candidate_ready"] = bool(shadow_available())
        data["shadow_candidate"] = shadow_metadata()
        data["shadow_candidate_sheet"] = "MLB_Candidate_Picks"
        return data

    def _active_enrichment(self):
        return getattr(self._enrichment_local, "value", None)

    def _current_offensive_index(self, team, fallback=100.0):
        baseline = super()._current_offensive_index(team, fallback)
        if self.enrichment_mode != "validated":
            return baseline
        ctx = self._active_enrichment() or {}; inputs = ctx.get("candidate_inputs", {})
        if team == ctx.get("home"): return float(inputs.get("off_home", baseline))
        if team == ctx.get("away"): return float(inputs.get("off_away", baseline))
        return baseline

    def _starter_metric(self, name):
        baseline = super()._starter_metric(name)
        if self.enrichment_mode != "validated": return baseline
        ctx = self._active_enrichment() or {}
        if str(name) == str(ctx.get("home_pitcher")):
            return float(ctx.get("candidate_inputs", {}).get("pitch_home") or baseline) if baseline is not None else ctx.get("candidate_inputs", {}).get("pitch_home")
        if str(name) == str(ctx.get("away_pitcher")):
            return float(ctx.get("candidate_inputs", {}).get("pitch_away") or baseline) if baseline is not None else ctx.get("candidate_inputs", {}).get("pitch_away")
        return baseline

    def _prepare_enrichment(self, game):
        home_name, away_name = game.get("home"), game.get("away")
        home, away = EQUIPOS_MAP.get(home_name, ""), EQUIPOS_MAP.get(away_name, "")
        base_off_home = MLBWebService._current_offensive_index(self, home) if home else 100.0
        base_off_away = MLBWebService._current_offensive_index(self, away) if away else 100.0
        base_pitch_home = MLBWebService._starter_metric(self, game.get("home_pitcher")) if home else None
        base_pitch_away = MLBWebService._starter_metric(self, game.get("away_pitcher")) if away else None
        if base_pitch_home is None and home: base_pitch_home = MLBWebService._team_pitching(self, home)
        if base_pitch_away is None and away: base_pitch_away = MLBWebService._team_pitching(self, away)
        enrichment = build_live_enrichment(
            self.batting, self.pitchers, home=home, away=away,
            home_pitcher=game.get("home_pitcher"), away_pitcher=game.get("away_pitcher"),
            base_off_home=base_off_home, base_off_away=base_off_away,
            base_pitch_home=base_pitch_home or 4.10, base_pitch_away=base_pitch_away or 4.10,
        )
        enrichment.update({"home": home, "away": away, "home_pitcher": game.get("home_pitcher"),
                           "away_pitcher": game.get("away_pitcher"), "mode": self.enrichment_mode,
                           "affects_pick": self.enrichment_mode == "validated"})
        return enrichment

    def _evaluate_game(self, game):
        enrichment = self._prepare_enrichment(game); self._enrichment_local.value = enrichment
        try: result = super()._evaluate_game(game)
        finally: self._enrichment_local.value = None
        result.setdefault("context", {})["enrichment"] = enrichment
        for row in result.get("diagnostics", []):
            row["enrichment_mode"] = self.enrichment_mode
            row["starter_hand_home"] = enrichment.get("home_starter", {}).get("hand")
            row["starter_hand_away"] = enrichment.get("away_starter", {}).get("hand")
            row["starter_profile_coverage_home"] = enrichment.get("home_starter", {}).get("coverage", 0.0)
            row["starter_profile_coverage_away"] = enrichment.get("away_starter", {}).get("coverage", 0.0)
            row["platoon_used_home"] = bool(enrichment.get("home_offense_vs_hand", {}).get("used"))
            row["platoon_used_away"] = bool(enrichment.get("away_offense_vs_hand", {}).get("used"))
        for row in result.get("accepted", []): row["enrichment_mode"] = self.enrichment_mode
        return result

    def _pitcher_id(self, name):
        try:
            if not name or self.pitchers.empty or "Name" not in self.pitchers.columns or "PlayerID" not in self.pitchers.columns:
                return None
            names = self.pitchers["Name"].astype(str)
            m = self.pitchers[names.str.casefold() == str(name).casefold()]
            if m.empty:
                last = str(name).split()[-1].casefold(); m = self.pitchers[names.str.split().str[-1].str.casefold() == last]
                if m["Name"].nunique() != 1: return None
            v = pd.to_numeric(pd.Series([m.iloc[-1]["PlayerID"]]), errors="coerce").iloc[0]
            return None if pd.isna(v) else int(v)
        except Exception:
            return None

    def _evaluate_shadow(self, game, baseline_result):
        if not shadow_available(): return {"accepted": [], "diagnostics": [], "error": "artifact_unavailable"}
        home_name, away_name = game["home"], game["away"]
        h, a = EQUIPOS_MAP.get(home_name, ""), EQUIPOS_MAP.get(away_name, "")
        if not h or not a: return {"accepted": [], "diagnostics": [], "error": "team_normalization"}
        off_h = MLBWebService._current_offensive_index(self, h); off_a = MLBWebService._current_offensive_index(self, a)
        pit_h = MLBWebService._starter_metric(self, game.get("home_pitcher")) or MLBWebService._team_pitching(self, h)
        pit_a = MLBWebService._starter_metric(self, game.get("away_pitcher")) or MLBWebService._team_pitching(self, a)
        if pit_h is None or pit_a is None: return {"accepted": [], "diagnostics": [], "error": "pitching_insufficient"}
        g = dict(game); g["home_pitcher_id"] = self._pitcher_id(game.get("home_pitcher")); g["away_pitcher_id"] = self._pitcher_id(game.get("away_pitcher"))
        ml = shadow_predict(self, g, h, a, off_h, off_a, pit_h, pit_a, slate_date())
        if not ml: return {"accepted": [], "diagnostics": [], "error": "prediction_unavailable"}

        runs = baseline_result.get("monte_carlo", {}).get("runs", {})
        p_mc_h = baseline_result.get("monte_carlo", {}).get("moneyline", {}).get("Gana Local", 50.0)
        p_mc_a = baseline_result.get("monte_carlo", {}).get("moneyline", {}).get("Gana Visita", 50.0)
        line = float(game["linea_carreras"])
        p_ml_h, p_ml_a = ml["Probabilidad_Local"], ml["Probabilidad_Visita"]
        p_ml_over = estimate_ml_probability(ml.get("Proyeccion_Carreras", line), line, "over", ml.get("Sigma_Carreras"))
        p_ml_under = estimate_ml_probability(ml.get("Proyeccion_Carreras", line), line, "under", ml.get("Sigma_Carreras"))
        p_mc_over, p_mc_under = runs.get(f"Over {line}", 50.0), runs.get(f"Under {line}", 50.0)
        spread_h, spread_a = game.get("spread_loc"), game.get("spread_vis")
        p_ml_sp_h = estimate_ml_probability(ml.get("Proyeccion_Handicap_Local", 0), spread_h, "spread", ml.get("Sigma_Handicap")) if spread_h is not None else 50.0
        p_ml_sp_a = estimate_ml_probability(-ml.get("Proyeccion_Handicap_Local", 0), spread_a, "spread", ml.get("Sigma_Handicap")) if spread_a is not None else 50.0
        p_mc_sp_h = runs.get(f"Spread Local {float(spread_h):+.1f}", 50.0) if spread_h is not None else 50.0
        p_mc_sp_a = runs.get(f"Spread Visita {float(spread_a):+.1f}", 50.0) if spread_a is not None else 50.0
        nv_h, nv_a = no_vig_two_way(game.get("cuota_loc"), game.get("cuota_vis")); nv_over, nv_under = no_vig_two_way(game.get("cuota_over"), game.get("cuota_under")); nv_sp_h, nv_sp_a = no_vig_two_way(game.get("cuota_spread_loc"), game.get("cuota_spread_vis"))
        candidates = [(moneyline_candidate(f"Gana Local ({home_name})", p_ml_h, p_mc_h, game.get("cuota_loc"), nv_h), None),
                      (moneyline_candidate(f"Gana Visita ({away_name})", p_ml_a, p_mc_a, game.get("cuota_vis"), nv_a), None)]
        if game.get("cuota_over") is not None: candidates.append((total_candidate(f"Over {line}", p_ml_over, p_mc_over, game.get("cuota_over"), nv_over, runs.get(f"Push {line}", 0.0)), line))
        if game.get("cuota_under") is not None: candidates.append((total_candidate(f"Under {line}", p_ml_under, p_mc_under, game.get("cuota_under"), nv_under, runs.get(f"Push {line}", 0.0)), line))
        if spread_h is not None and game.get("cuota_spread_loc") is not None: candidates.append((runline_candidate(f"Hándicap {float(spread_h):+.1f} ({home_name})", p_ml_sp_h, p_mc_sp_h, game.get("cuota_spread_loc"), nv_sp_h, runs.get(f"Push Spread Local {float(spread_h):+.1f}", 0.0)), float(spread_h)))
        if spread_a is not None and game.get("cuota_spread_vis") is not None: candidates.append((runline_candidate(f"Hándicap {float(spread_a):+.1f} ({away_name})", p_ml_sp_a, p_mc_sp_a, game.get("cuota_spread_vis"), nv_sp_a, runs.get(f"Push Spread Visita {float(spread_a):+.1f}", 0.0)), float(spread_a)))
        accepted, diagnostics = [], []
        for cand, market_line in candidates:
            if cand is None: continue
            row = {"game_pk": game.get("game_pk"), "partido": f"{away_name} @ {home_name}", "mercado": cand.market,
                   "apuesta": cand.selection, "linea": market_line, "prob_ml": round(cand.prob_ml,2), "prob_mc": round(cand.prob_mc,2),
                   "probabilidad": round(cand.probability,2), "cuota": cand.odds, "no_vig": cand.market_no_vig,
                   "edge_pp": cand.edge_pp, "ev_pct": cand.ev_pct, "desacuerdo_pp": cand.disagreement_pp, "score": cand.score,
                   "push_probability": cand.push_probability, "kelly_pct": kelly_fraction_pct(cand.probability,cand.odds,push_probability=cand.push_probability),
                   "accepted": bool(cand.accepted), "reason": cand.reason, "candidate_version": ml.get("Model_Version"), "day_night": ml.get("DayNight")}
            diagnostics.append(row)
            if cand.accepted: accepted.append(row)
        return {"accepted": accepted, "diagnostics": diagnostics, "ml": ml, "error": None}

    def scan(self, persist=True):
        if not self.model_ready: raise RuntimeError("Modelo ML no disponible")
        games = self.slate(); accepted = []; diagnostics = []; shadow_accepted = []; shadow_diagnostics = []; errors = []
        for game in games:
            try:
                result = self._evaluate_game(game); accepted.extend(result["accepted"]); diagnostics.extend(result["diagnostics"])
                shadow = self._evaluate_shadow(game, result); shadow_accepted.extend(shadow.get("accepted", [])); shadow_diagnostics.extend(shadow.get("diagnostics", []))
                if shadow.get("error") and shadow.get("error") != "artifact_unavailable": errors.append({"game_pk":game.get("game_pk"),"partido":f"{game.get('away')} @ {game.get('home')}","shadow_error":shadow.get("error")})
            except Exception as exc:
                errors.append({"game_pk": game.get("game_pk"), "partido": f"{game.get('away')} @ {game.get('home')}", "error": str(exc)[:200]})
        accepted = sorted(accepted, key=lambda r: float(r.get("score", -999)), reverse=True)[:3]
        shadow_accepted = sorted(shadow_accepted, key=lambda r: float(r.get("score", -999)), reverse=True)[:3]
        shadow_sheet_status = None
        if persist and accepted:
            rows=[]
            for r in accepted:
                rows.append({'game_date':slate_date().isoformat(),'game_pk':r['game_pk'],'away':r['partido'].split(' @ ')[0],'home':r['partido'].split(' @ ')[1],
                             'market':r['mercado'],'selection':r['apuesta'],'line':r['linea'],'odds':r['cuota'],'prob_ml':r['prob_ml'],'prob_mc':r['prob_mc'],
                             'prob_combined':r['probabilidad'],'market_no_vig':r['no_vig'],'edge_pp':r['edge_pp'],'ev_pct':r['ev_pct'],'disagreement_pp':r['desacuerdo_pp'],
                             'score':r['score'],'model_version':'v7-cloudrun','result_status':'pending'})
            try: append_snapshot(rows)
            except Exception: pass
        if persist and shadow_accepted:
            rows=[]
            for r in shadow_accepted:
                rows.append({'game_date':slate_date().isoformat(),'game_pk':r['game_pk'],'away':r['partido'].split(' @ ')[0],'home':r['partido'].split(' @ ')[1],
                             'market':r['mercado'],'selection':r['apuesta'],'line':r['linea'],'odds':r['cuota'],'prob_ml':r['prob_ml'],'prob_mc':r['prob_mc'],
                             'prob_combined':r['probabilidad'],'market_no_vig':r['no_vig'],'edge_pp':r['edge_pp'],'ev_pct':r['ev_pct'],'disagreement_pp':r['desacuerdo_pp'],
                             'score':r['score'],'model_version':r.get('candidate_version') or 'shadow-candidate-v1','result_status':'pending','kelly_pct':r.get('kelly_pct')})
            shadow_sheet_status = append_shadow_snapshot(rows)
        return {"date":slate_date().isoformat(),"recommendations":accepted,"diagnostics":sorted(diagnostics,key=lambda r:float(r.get('score',-999)),reverse=True)[:12],
                "shadow_candidate":{"ready":bool(shadow_available()),"recommendations":shadow_accepted,"diagnostics":sorted(shadow_diagnostics,key=lambda r:float(r.get('score',-999)),reverse=True)[:12],"sheet_sync":shadow_sheet_status},
                "errors":errors,"games_seen":len(games),"model":self.health()}
