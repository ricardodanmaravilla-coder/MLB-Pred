import json

import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error

from modules.ml_mlb import PredictorMLMLB
from modules.team_utils import normalize_team


def main():
    bat = pd.read_csv('data/mlb_batting.csv')
    pit = pd.read_csv('data/mlb_pitching.csv')
    games = pd.read_csv('data/mlb_games.csv')
    games['Date'] = pd.to_datetime(games['Date'], errors='coerce')
    games['Season'] = pd.to_numeric(games['Season'], errors='coerce')
    games = games.dropna(subset=['Date','Season','Home_Score','Away_Score']).sort_values('Date').reset_index(drop=True)

    # Holdout cronológico: último 15% de los juegos.
    cut = int(len(games) * 0.85)
    train = games.iloc[:cut].copy()
    test = games.iloc[cut:].copy()

    model = PredictorMLMLB()
    if not model.entrenar(bat, pit, train):
        raise RuntimeError('No se pudo entrenar el modelo')

    bat2 = bat.copy(); pit2 = pit.copy()
    bat2['TeamKey'] = bat2['Team'].map(normalize_team); pit2['TeamKey'] = pit2['Team'].map(normalize_team)
    bat2['Season'] = pd.to_numeric(bat2['Season'], errors='coerce'); pit2['Season'] = pd.to_numeric(pit2['Season'], errors='coerce')
    bat2['wRC+'] = pd.to_numeric(bat2['wRC+'], errors='coerce'); pit2['xFIP'] = pd.to_numeric(pit2['xFIP'], errors='coerce')
    bd = bat2.dropna(subset=['TeamKey','Season','wRC+']).set_index(['TeamKey','Season'])['wRC+'].to_dict()
    pdict = pit2.dropna(subset=['TeamKey','Season','xFIP']).set_index(['TeamKey','Season'])['xFIP'].to_dict()
    bmed = float(bat2['wRC+'].median()); pmed = float(pit2['xFIP'].median())

    y, probs, runs_true, runs_pred = [], [], [], []
    for _, r in test.iterrows():
        h, a, season = normalize_team(r['Home']), normalize_team(r['Away']), int(r['Season'])
        sy = season - 1
        wh = float(bd.get((h, sy), bmed)); wa = float(bd.get((a, sy), bmed))
        ph = float(pdict.get((h, sy), pmed)); pa = float(pdict.get((a, sy), pmed))
        pred = model.predecir_partido(h, a, wh, wa, ph, pa)
        probs.append(pred['Probabilidad_Local'] / 100.0)
        y.append(int(float(r['Home_Score']) > float(r['Away_Score'])))
        runs_true.append(float(r['Home_Score']) + float(r['Away_Score']))
        runs_pred.append(float(pred['Proyeccion_Carreras']))

    picks = [int(p >= 0.5) for p in probs]
    base_p = sum(y) / len(y)
    baseline_probs = [base_p] * len(y)
    result = {
        'n_train': len(train),
        'n_test': len(test),
        'accuracy': round(accuracy_score(y, picks), 4),
        'baseline_accuracy_home_rate': round(max(base_p, 1-base_p), 4),
        'brier': round(brier_score_loss(y, probs), 4),
        'baseline_brier': round(brier_score_loss(y, baseline_probs), 4),
        'logloss': round(log_loss(y, probs, labels=[0,1]), 4),
        'baseline_logloss': round(log_loss(y, baseline_probs, labels=[0,1]), 4),
        'runs_mae': round(mean_absolute_error(runs_true, runs_pred), 3),
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
