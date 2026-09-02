"""Populate exact historical starter identity into the isolated research input.

This script is intended for the experimental Parquet workflow only. It downloads
MLB schedule starter identities, then enriches the workflow checkout's
``data/mlb_games.csv`` by GameID/gamePk. Production Cloud Run does not call it.
"""
from pathlib import Path

import pandas as pd

import build_game_starters_history as starter_builder

GAMES = Path("data/mlb_games.csv")
STARTERS = Path("data/mlb_game_starters_history.csv")


def main() -> None:
    starter_builder.main()
    games = pd.read_csv(GAMES, low_memory=False)
    starters = pd.read_csv(STARTERS, low_memory=False)

    game_key = "GameID" if "GameID" in games.columns else "gamePk" if "gamePk" in games.columns else None
    if game_key is None:
        raise RuntimeError("mlb_games.csv no tiene GameID ni gamePk")

    games[game_key] = pd.to_numeric(games[game_key], errors="coerce")
    starters["GameID"] = pd.to_numeric(starters["GameID"], errors="coerce")
    keep = ["GameID", "HomeStarterID", "HomeStarterName", "AwayStarterID", "AwayStarterName"]
    enriched = games.merge(starters[keep], left_on=game_key, right_on="GameID", how="left", suffixes=("", "_starter_map"))

    if game_key != "GameID":
        enriched = enriched.drop(columns=["GameID_starter_map"], errors="ignore")
    for col in ("HomeStarterID", "HomeStarterName", "AwayStarterID", "AwayStarterName"):
        mapped = f"{col}_starter_map"
        if mapped in enriched.columns:
            if col in games.columns:
                enriched[col] = enriched[col].where(enriched[col].notna(), enriched[mapped])
            else:
                enriched[col] = enriched[mapped]
            enriched = enriched.drop(columns=[mapped])

    both = (enriched["HomeStarterID"].notna() & enriched["AwayStarterID"].notna()).mean()
    if both < 0.50:
        raise RuntimeError(f"Cobertura histórica de ambos abridores demasiado baja: {both:.1%}")

    enriched.to_csv(GAMES, index=False)
    print(f"Experimental games enriched: rows={len(enriched)} both_starters={both:.1%}")


if __name__ == "__main__":
    main()
