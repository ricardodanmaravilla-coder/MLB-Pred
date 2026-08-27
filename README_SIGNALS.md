# MLB-Pred V7 signal architecture

The system separates **captured**, **eligible**, and **promoted** signals. Capturing a metric never forces it into production.

## Leak-safe ML candidates

Historical ML candidates are built from information available before the game. Team-season metrics use **season-1** and walk-forward predictions receive the game date explicitly. Current candidates include:

- Offense quality: wOBA, ISO, BB%, K%, EV, HardHit%, Barrel%.
- Platoon profile: OPS/OBP/SLG vs L and vs R from MLB StatsAPI `statSplits`.
- Run prevention: FIP, xFIP, SIERA, WHIP, K-BB%, GB%, HR/9.
- Schedule context: home/away rest days.
- Existing Big Data form features: recent 5/20-game production/prevention and shrunk H2H.

Missing values are neutral, never fabricated. The model trains `baseline20` and `advanced_prior_season` on the same complete-date validation split. Advanced features are promoted only when the composite Brier/runs-MAE/diff-MAE score improves and no target suffers material degradation. Coverage of historical metrics must also pass the configured gate.

## Live contextual engine

Monte Carlo/scanner continues to use signals that are known pregame but do not yet have an equivalent historical pregame archive for every game:

- Confirmed starter run prevention (real xFIP/FIP/SIERA when available, otherwise tagged fallback).
- Starter `PlayerID` and `PitchHand` are persisted for exact future platoon matching.
- Current bullpen quality plus a conservative completed-schedule workload/rest adjustment capped to a small range.
- Park factor, altitude, temperature, wind direction/speed.
- Current offensive index and live sportsbook market prices.
- Negative-binomial run dispersion estimated from recent completed MLB history.

## Data provenance and coverage

`minero_mlb.py` attempts FanGraphs/Statcast enrichment and official MLB L/R splits. `minero_pitchers.py` persists pitcher IDs/hands and expanded pitching metrics. `python signal_audit.py` reports coverage; the daily workflow warns when too few historical advanced metrics reach 65% coverage.

## Safety rules

- No same-day result may feed another game on that date.
- Backtests pass each game date explicitly through prediction and result updates.
- Season-level historical signals always use season-1.
- Missing/low-coverage signals do not force advanced activation.
- A new signal must improve leak-safe validation before promotion.
- Picks still require complete two-way no-vig reference, EV, edge and ML/MC agreement.
- Without historical sportsbook odds the project does not claim historical ROI.
