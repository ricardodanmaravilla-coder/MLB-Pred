# MLB-Pred advanced signals

Production separates signals into two groups.

## Leak-safe ML signals

These may enter the ML model only as prior-season features and only when coverage and chronological validation justify them: wOBA, ISO, BB%, K%, EV, HardHit%, Barrel%, FIP, xFIP, SIERA, WHIP, K-BB%, GB%, HR/9. Missing values are neutral, never fabricated. The model trains a baseline and an advanced candidate on the same date-safe validation split. Advanced signals are promoted only when the composite validation score improves without material harm to any target.

## Live contextual signals

Confirmed starter quality, current bullpen, park factor, altitude, temperature, wind and market prices remain in the live Monte Carlo/scanner path. They are not retroactively inserted into historical ML unless an equivalent pregame historical source exists.

## Safety rules

- No same-day result may feed another game on that date.
- Historical features use season-1 statistics.
- Coverage is reported explicitly by `python signal_audit.py`.
- Missing or low-coverage signals do not force model activation.
- Market recommendations still require complete two-way no-vig reference, EV, edge and ML/MC agreement.
