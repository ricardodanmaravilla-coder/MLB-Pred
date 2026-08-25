import math
import pandas as pd

from modules.historical_mlb import prepare_games
from modules.pick_ledger import LEDGER_COLUMNS
from modules.team_utils import normalize_team

LEDGER = 'data/picks_ledger.csv'
GAMES = 'data/mlb_games.csv'


def _f(v, default=None):
    try:
        if pd.isna(v):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def settle_row(row, game):
    hs = float(game.Home_Score)
    aw = float(game.Away_Score)
    market = str(row.get('market') or '')
    selection = str(row.get('selection') or '')
    line = _f(row.get('line'))
    odds = _f(row.get('odds'))
    if odds is None or odds <= 1:
        return None

    result = None
    if market == 'Moneyline':
        selected_home = 'Local' in selection or str(row.get('home')) in selection
        won = hs > aw if selected_home else aw > hs
        result = 'win' if won else 'loss'
    elif market == 'Totales' and line is not None:
        total = hs + aw
        if math.isclose(total, line):
            result = 'push'
        elif selection.strip().lower().startswith('over'):
            result = 'win' if total > line else 'loss'
        else:
            result = 'win' if total < line else 'loss'
    elif market == 'Hándicap' and line is not None:
        selected_home = str(row.get('home')) in selection
        margin = (hs - aw + line) if selected_home else (aw - hs + line)
        if math.isclose(margin, 0.0):
            result = 'push'
        else:
            result = 'win' if margin > 0 else 'loss'
    if result is None:
        return None
    profit = round(odds - 1.0, 4) if result == 'win' else (-1.0 if result == 'loss' else 0.0)
    return result, profit, f'{int(hs)}-{int(aw)}'


def main():
    try:
        ledger = pd.read_csv(LEDGER)
    except Exception:
        print('No ledger to settle')
        return
    if ledger.empty:
        print('Ledger empty')
        return
    games = prepare_games(pd.read_csv(GAMES))
    if games.empty:
        raise RuntimeError('No completed games available')
    games = games.copy()
    games['_date'] = games['Date'].dt.date.astype(str)
    games['_h'] = games['Home'].map(normalize_team)
    games['_a'] = games['Away'].map(normalize_team)
    if 'GameID' in games.columns:
        games['_gid'] = pd.to_numeric(games['GameID'], errors='coerce')

    settled = 0
    for idx, row in ledger.iterrows():
        if str(row.get('result_status') or 'pending') != 'pending':
            continue
        game_pk = _f(row.get('game_pk'))
        matches = pd.DataFrame()
        if game_pk is not None and '_gid' in games.columns:
            matches = games[games['_gid'] == game_pk]
        if matches.empty:
            h, a = normalize_team(row.get('home')), normalize_team(row.get('away'))
            d = str(row.get('game_date'))
            matches = games[(games['_date'] == d) & (games['_h'] == h) & (games['_a'] == a)]
        if len(matches) != 1:
            # Old rows without game_pk remain pending when a same-day matchup is ambiguous.
            continue
        settled_value = settle_row(row, matches.iloc[0])
        if settled_value is None:
            continue
        status, profit, score = settled_value
        ledger.at[idx, 'result_status'] = status
        ledger.at[idx, 'profit_units'] = profit
        ledger.at[idx, 'result_value'] = score
        settled += 1

    for c in LEDGER_COLUMNS:
        if c not in ledger.columns:
            ledger[c] = None
    ledger[LEDGER_COLUMNS].to_csv(LEDGER, index=False)
    print(f'Settled picks: {settled}')


if __name__ == '__main__':
    main()
