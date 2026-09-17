# MLB Probability Lab v1

Experimental branch only. V7 production code is unchanged.

## Rule
- Same V7 evaluation structure and inputs.
- A pick qualifies only when `prob_ml >= 60` and `prob_mc >= 60` for that candidate.
- EV, edge and no-vig are retained only as diagnostics; they do not accept/reject picks.
- All qualifying picks are retained.

## Stake
- Minimum: 3% of bankroll at 60% joint probability.
- Linear increase using `(prob_ml + prob_mc) / 2`.
- Maximum: 10% at 80% joint probability and above.

## Isolation
- Standalone runner: `python run_probability_lab.py`.
- Dedicated worksheet: `MLB_Probability_Lab`.
- Dedicated model/filter labels: `v7-probability-lab-v1` / `probability-only-60-60-v1`.
- The lab never calls the V7 production `append_snapshot` writer.
- No V7 source file is modified by this branch.
