"""Big Data layer for MLB-Pred.

DuckDB-backed local warehouse + Parquet feature store. The design is intentionally
optional and additive: if the warehouse is not built, the production Streamlit app
continues using the existing CSV pipeline unchanged.

The core invariant is temporal safety: every pregame feature for date D is computed
only from games strictly before D.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from .historical_mlb import append_game, h2h_state, prepare_games, team_state
from .team_utils import normalize_team

try:
    import duckdb
except Exception:  # pragma: no cover - production remains backward compatible
    duckdb = None


FEATURE_VERSION = "bd_v1"
FEATURE_COLUMNS = [
    "home_win5", "away_win5", "home_win20", "away_win20",
    "home_rf5", "away_rf5", "home_ra5", "away_ra5",
    "home_rd5", "away_rd5", "home_rd20", "away_rd20",
    "h2h_home_win", "h2h_home_rd", "h2h_sample",
    "home_rest_days", "away_rest_days",
]


@dataclass(frozen=True)
class BigDataPaths:
    root: Path = Path("data/bigdata")

    @property
    def db(self) -> Path:
        return self.root / "mlb.duckdb"

    @property
    def features(self) -> Path:
        return self.root / f"pregame_features_{FEATURE_VERSION}.parquet"


class MLBDataWarehouse:
    """Persistent analytical store for historical games and leak-safe features."""

    def __init__(self, root: str | Path = "data/bigdata"):
        self.paths = BigDataPaths(Path(root))
        self.paths.root.mkdir(parents=True, exist_ok=True)
        if duckdb is None:
            raise RuntimeError("duckdb no esta instalado; ejecuta pip install -r requirements.txt")

    def connect(self):
        return duckdb.connect(str(self.paths.db))

    @staticmethod
    def _normalize_games(df_games: pd.DataFrame) -> pd.DataFrame:
        g = prepare_games(df_games)
        if g.empty:
            return g
        keep = [c for c in [
            "Date", "Season", "GameType", "Home", "Away", "Home_Score", "Away_Score",
            "gamePk", "Venue", "DayNight", "TempF", "WindMph", "AltitudeFt", "ParkFactor"
        ] if c in g.columns]
        g = g[keep].copy()
        g["Home"] = g["Home"].map(normalize_team)
        g["Away"] = g["Away"].map(normalize_team)
        g["Date"] = pd.to_datetime(g["Date"], errors="coerce")
        g = g.dropna(subset=["Date", "Home", "Away", "Home_Score", "Away_Score"])
        g["game_key"] = (
            g["Date"].dt.strftime("%Y-%m-%d") + "|" + g["Away"].astype(str) + "@" + g["Home"].astype(str)
        )
        if "gamePk" in g.columns:
            pk = pd.to_numeric(g["gamePk"], errors="coerce")
            g.loc[pk.notna(), "game_key"] = "pk:" + pk[pk.notna()].astype("int64").astype(str)
        return g.sort_values(["Date", "game_key"]).drop_duplicates("game_key", keep="last").reset_index(drop=True)

    def ingest_games(self, df_games: pd.DataFrame) -> int:
        g = self._normalize_games(df_games)
        if g.empty:
            return 0
        con = self.connect()
        try:
            con.register("incoming_games", g)
            con.execute("CREATE TABLE IF NOT EXISTS games AS SELECT * FROM incoming_games WHERE 1=0")
            # Schema can grow over time without invalidating old warehouse files.
            existing = {r[1] for r in con.execute("PRAGMA table_info('games')").fetchall()}
            for c in g.columns:
                if c not in existing:
                    con.execute(f'ALTER TABLE games ADD COLUMN "{c}" VARCHAR')
            con.execute("DELETE FROM games USING incoming_games WHERE games.game_key = incoming_games.game_key")
            common = [c for c in g.columns if c in {r[1] for r in con.execute("PRAGMA table_info('games')").fetchall()}]
            cols = ",".join(f'"{c}"' for c in common)
            con.execute(f"INSERT INTO games ({cols}) SELECT {cols} FROM incoming_games")
            con.execute("CREATE INDEX IF NOT EXISTS idx_games_date ON games(Date)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_games_key ON games(game_key)")
            return len(g)
        finally:
            con.close()

    @staticmethod
    def build_leak_safe_features(df_games: pd.DataFrame) -> pd.DataFrame:
        games = MLBDataWarehouse._normalize_games(df_games)
        if games.empty:
            return pd.DataFrame()

        hist: Dict[str, list] = {}
        hh: Dict[Tuple[str, str], list] = {}
        last_game: Dict[str, pd.Timestamp] = {}
        rows = []

        for r in games.itertuples(index=False):
            loc, vis = str(r.Home), str(r.Away)
            date = pd.Timestamp(r.Date)
            w5l, rf5l, ra5l, rd5l = team_state(hist, loc, 5)
            w5v, rf5v, ra5v, rd5v = team_state(hist, vis, 5)
            w20l, _, _, rd20l = team_state(hist, loc, 20)
            w20v, _, _, rd20v = team_state(hist, vis, 20)
            hwin, hrd, hn = h2h_state(hh, loc, vis, 12)

            def rest(team: str) -> float:
                if team not in last_game:
                    return 3.0
                # 0 = played previous calendar day; clip long layoffs.
                return float(np.clip((date.normalize() - last_game[team].normalize()).days - 1, 0, 7))

            rows.append({
                "game_key": r.game_key, "Date": date, "Season": int(r.Season),
                "Home": loc, "Away": vis,
                "home_win5": w5l, "away_win5": w5v,
                "home_win20": w20l, "away_win20": w20v,
                "home_rf5": rf5l, "away_rf5": rf5v,
                "home_ra5": ra5l, "away_ra5": ra5v,
                "home_rd5": rd5l, "away_rd5": rd5v,
                "home_rd20": rd20l, "away_rd20": rd20v,
                "h2h_home_win": hwin, "h2h_home_rd": hrd,
                "h2h_sample": min(hn, 12) / 12.0,
                "home_rest_days": rest(loc), "away_rest_days": rest(vis),
                "target_home_win": int(float(r.Home_Score) > float(r.Away_Score)),
                "target_total_runs": float(r.Home_Score) + float(r.Away_Score),
                "target_run_diff": float(r.Home_Score) - float(r.Away_Score),
                "feature_version": FEATURE_VERSION,
            })

            append_game(hist, hh, loc, vis, float(r.Home_Score), float(r.Away_Score))
            last_game[loc] = date
            last_game[vis] = date

        return pd.DataFrame(rows)

    def rebuild_feature_store(self) -> int:
        con = self.connect()
        try:
            games = con.execute("SELECT * FROM games ORDER BY Date, game_key").df()
        finally:
            con.close()
        features = self.build_leak_safe_features(games)
        if features.empty:
            return 0
        features.to_parquet(self.paths.features, index=False)
        con = self.connect()
        try:
            con.register("feature_frame", features)
            con.execute("DROP TABLE IF EXISTS pregame_features")
            con.execute("CREATE TABLE pregame_features AS SELECT * FROM feature_frame")
            con.execute("CREATE INDEX IF NOT EXISTS idx_features_date ON pregame_features(Date)")
        finally:
            con.close()
        return len(features)

    def training_frame(self, cutoff_date: Optional[str | pd.Timestamp] = None) -> pd.DataFrame:
        """Return model-ready observations; cutoff is exclusive to prevent leakage."""
        con = self.connect()
        try:
            if cutoff_date is None:
                return con.execute("SELECT * FROM pregame_features ORDER BY Date, game_key").df()
            cutoff = pd.Timestamp(cutoff_date)
            return con.execute(
                "SELECT * FROM pregame_features WHERE Date < ? ORDER BY Date, game_key", [cutoff]
            ).df()
        finally:
            con.close()

    def status(self) -> dict:
        con = self.connect()
        try:
            def count(table: str) -> int:
                try:
                    return int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                except Exception:
                    return 0
            return {
                "backend": "duckdb",
                "db": str(self.paths.db),
                "feature_version": FEATURE_VERSION,
                "games": count("games"),
                "features": count("pregame_features"),
            }
        finally:
            con.close()


def bootstrap_from_repository(data_dir: str | Path = "data", root: str | Path = "data/bigdata") -> dict:
    data_dir = Path(data_dir)
    games_path = data_dir / "mlb_games.csv"
    if not games_path.exists():
        raise FileNotFoundError(games_path)
    wh = MLBDataWarehouse(root)
    games = pd.read_csv(games_path)
    ingested = wh.ingest_games(games)
    features = wh.rebuild_feature_store()
    out = wh.status()
    out.update({"ingested": ingested, "rebuilt_features": features})
    return out
