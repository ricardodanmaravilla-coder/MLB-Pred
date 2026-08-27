import numpy as np
import pandas as pd

from .team_utils import normalize_team


def prepare_games(df_games):
    """Normalize completed historical results without leaking exhibition games.

    Legacy rows have no GameType. When newer rows begin carrying GameType, preserve
    those legacy rows using the old month heuristic instead of dropping the entire
    historical corpus merely because the column now exists.

    The historical miner stores MLB's unique identifier as ``GameID``. Big Data
    historically expected ``gamePk`` instead, so expose the same identifier under
    both names when needed. This prevents same-day doubleheaders from collapsing
    into a single date/team key and gives chronological consumers a deterministic
    secondary ordering key.
    """
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

    # MLB-StatsAPI's statsapi.schedule() is persisted by minero_mlb.py as GameID.
    # The warehouse uses gamePk as its stable identifier. Preserve both contracts.
    if 'GameID' in g.columns:
        g['GameID'] = pd.to_numeric(g['GameID'], errors='coerce')
        if 'gamePk' not in g.columns:
            g['gamePk'] = g['GameID']
        else:
            existing_pk = pd.to_numeric(g['gamePk'], errors='coerce')
            g['gamePk'] = existing_pk.where(existing_pk.notna(), g['GameID'])
    elif 'gamePk' in g.columns:
        g['gamePk'] = pd.to_numeric(g['gamePk'], errors='coerce')

    g = g.dropna(subset=['Date','Home','Away','Home_Score','Away_Score','Season'])

    if 'GameType' in g.columns:
        gt = g['GameType']
        known = gt.notna() & gt.astype(str).str.strip().ne('')
        competitive_known = known & gt.astype(str).isin(['R','P'])
        legacy_unknown = (~known) & (g['Date'].dt.month >= 4)
        g = g[competitive_known | legacy_unknown]
    else:
        g = g[g['Date'].dt.month >= 4]

    sort_cols = ['Date']
    if 'gamePk' in g.columns:
        sort_cols.append('gamePk')
    return g.sort_values(sort_cols, kind='mergesort').reset_index(drop=True)


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
    weight = count / (count + 10.0)
    return 0.5 + (wins-0.5)*weight, rd*weight, count


def append_game(history, history_h2h, loc, vis, home_score, away_score):
    hw = int(home_score > away_score)
    aw = int(away_score > home_score)
    history.setdefault(loc, []).append((hw, home_score, away_score))
    history.setdefault(vis, []).append((aw, away_score, home_score))
    history_h2h.setdefault((loc, vis), []).append((hw, home_score-away_score))
    history_h2h.setdefault((vis, loc), []).append((aw, away_score-home_score))
