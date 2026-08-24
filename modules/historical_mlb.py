import numpy as np
import pandas as pd

from .team_utils import normalize_team


def prepare_games(df_games):
    """Normalize historical results and exclude non-competitive spring games when possible."""
    if df_games is None or df_games.empty:
        return pd.DataFrame()
    g = df_games.copy()
    g['Date'] = pd.to_datetime(g.get('Date'), errors='coerce')
    g['Home'] = g['Home'].map(normalize_team)
    g['Away'] = g['Away'].map(normalize_team)
    g['Home_Score'] = pd.to_numeric(g['Home_Score'], errors='coerce')
    g['Away_Score'] = pd.to_numeric(g['Away_Score'], errors='coerce')
    if 'Season' not in g.columns:
        g['Season'] = g['Date'].dt.year
    g['Season'] = pd.to_numeric(g['Season'], errors='coerce')
    g = g.dropna(subset=['Date','Home','Away','Home_Score','Away_Score','Season'])
    if 'GameType' in g.columns and g['GameType'].notna().any():
        g = g[g['GameType'].astype(str).isin(['R','P'])]
    else:
        # Legacy CSV has no game type. February/March rows are overwhelmingly spring training;
        # exclude them from model training rather than treating exhibitions as MLB regular games.
        g = g[g['Date'].dt.month >= 4]
    return g.sort_values('Date').reset_index(drop=True)


def team_state(history, team, n=20):
    rows = history.get(team, [])[-int(n):]
    if not rows:
        return 0.5, 4.5, 4.5, 0.0
    wins = float(np.mean([r[0] for r in rows]))
    rf = float(np.mean([r[1] for r in rows]))
    ra = float(np.mean([r[2] for r in rows]))
    return wins, rf, ra, rf-ra


def h2h_state(history_h2h, loc, vis, n=12):
    rows = history_h2h.get((loc, vis), [])[-int(n):]
    if not rows:
        return 0.5, 0.0, 0
    wins = float(np.mean([r[0] for r in rows]))
    rd = float(np.mean([r[1] for r in rows]))
    count = len(rows)
    # Empirical Bayes shrinkage: small H2H samples stay close to neutral.
    weight = count / (count + 10.0)
    return 0.5 + (wins-0.5)*weight, rd*weight, count


def append_game(history, history_h2h, loc, vis, home_score, away_score):
    hw = int(home_score > away_score)
    aw = int(away_score > home_score)
    history.setdefault(loc, []).append((hw, home_score, away_score))
    history.setdefault(vis, []).append((aw, away_score, home_score))
    history_h2h.setdefault((loc, vis), []).append((hw, home_score-away_score))
    history_h2h.setdefault((vis, loc), []).append((aw, away_score-home_score))
