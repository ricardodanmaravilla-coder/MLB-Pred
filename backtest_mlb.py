import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error, mean_squared_error

from modules.ml_mlb import PredictorMLMLB


def walk_forward(batting_path, pitching_path, games_path, min_train=5000, step=1500):
    """Walk-forward cronológico por bloques amplios.

    Reduce tiempo de CI sin mezclar futuro/pasado: cada bloque se predice con un
    modelo entrenado exclusivamente en juegos anteriores.
    """
    bat = pd.read_csv(batting_path)
    pit = pd.read_csv(pitching_path)
    games = pd.read_csv(games_path)
    builder = PredictorMLMLB()
    prep = builder.preparar_dataset(bat, pit, games)
    rows = []

    for end in range(min_train, len(prep), step):
        train = prep.iloc[:end].copy()
        test = prep.iloc[end:min(end + step, len(prep))].copy()
        model = PredictorMLMLB()
        if not model.entrenar_preparado(train):
            continue
        X = test[model.FEATURES].astype(float)
        p_home = model.modelo_ganador.predict_proba(X)[:, 1]
        runs = model.modelo_carreras.predict(X)
        diff = model.modelo_handicap.predict(X)
        for i, (_, r) in enumerate(test.iterrows()):
            rows.append({
                "Date": r.Date, "Home": r.Home, "Away": r.Away,
                "y_home": int(r.Target_Win), "p_home": float(p_home[i]),
                "actual_runs": float(r.Target_Runs), "pred_runs": float(runs[i]),
                "actual_diff": float(r.Target_Diff), "pred_diff": float(diff[i]),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out, {}
    y = out.y_home.to_numpy(int)
    p = out.p_home.to_numpy(float)
    metrics = {
        "n_oos": int(len(out)),
        "accuracy_moneyline": float(accuracy_score(y, p >= 0.5)),
        "brier_moneyline": float(brier_score_loss(y, p)),
        "logloss_moneyline": float(log_loss(y, np.column_stack([1-p, p]), labels=[0, 1])),
        "mae_total_runs": float(mean_absolute_error(out.actual_runs, out.pred_runs)),
        "rmse_total_runs": float(mean_squared_error(out.actual_runs, out.pred_runs) ** 0.5),
        "mae_run_diff": float(mean_absolute_error(out.actual_diff, out.pred_diff)),
    }
    for threshold in (0.55, 0.60, 0.65):
        confident = (p >= threshold) | (p <= 1-threshold)
        if confident.any():
            pred = (p[confident] >= 0.5).astype(int)
            metrics[f"n_conf_{int(threshold*100)}"] = int(confident.sum())
            metrics[f"acc_conf_{int(threshold*100)}"] = float(np.mean(pred == y[confident]))
    return out, metrics


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "data"
    rows, metrics = walk_forward(
        f"{base}/mlb_batting.csv", f"{base}/mlb_pitching.csv", f"{base}/mlb_games.csv"
    )
    print(metrics)
    rows.to_csv("backtest_mlb_predictions.csv", index=False)
