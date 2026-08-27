"""DuckDB/Parquet Big Data layer for MLB-Pred with leak-safe chronological features."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import hashlib
import json
import uuid

import numpy as np
import pandas as pd

from .historical_mlb import append_game, h2h_state, prepare_games, team_state
from .team_utils import normalize_team
from .metric_quality import batting_metric, pitching_metric

try:
    import duckdb
except Exception:  # pragma: no cover
    duckdb = None

FEATURE_VERSION = "bd_v2"
FEATURE_COLUMNS = [
    "home_win5","away_win5","home_win20","away_win20","home_rf5","away_rf5",
    "home_ra5","away_ra5","home_rd5","away_rd5","home_rd20","away_rd20",
    "h2h_home_win","h2h_home_rd","h2h_sample","home_rest_days","away_rest_days",
]
LEGACY_ML_COLUMNS = [
    "home_win5","away_win5","home_win20","away_win20","home_rf5","away_rf5",
    "home_ra5","away_ra5","home_rd5","away_rd5","home_rd20","away_rd20",
    "h2h_home_win","h2h_home_rd","h2h_sample","home_offense","away_offense",
    "home_pitching","away_pitching","home_field",
]

@dataclass(frozen=True)
class BigDataPaths:
    root: Path = Path("data/bigdata")
    @property
    def db(self): return self.root / "mlb.duckdb"
    @property
    def features(self): return self.root / f"pregame_features_{FEATURE_VERSION}.parquet"

class MLBDataWarehouse:
    def __init__(self, root: str | Path = "data/bigdata"):
        self.paths = BigDataPaths(Path(root)); self.paths.root.mkdir(parents=True, exist_ok=True)
        if duckdb is None: raise RuntimeError("duckdb no esta instalado; ejecuta pip install -r requirements.txt")

    def connect(self): return duckdb.connect(str(self.paths.db))

    @staticmethod
    def _normalize_games(df_games: pd.DataFrame) -> pd.DataFrame:
        g = prepare_games(df_games)
        if g.empty: return g
        keep = [c for c in ["Date","Season","GameType","Home","Away","Home_Score","Away_Score",
                            "gamePk","Venue","DayNight","TempF","WindMph","AltitudeFt","ParkFactor"] if c in g.columns]
        g = g[keep].copy(); g["Home"] = g["Home"].map(normalize_team); g["Away"] = g["Away"].map(normalize_team)
        g["Date"] = pd.to_datetime(g["Date"], errors="coerce")
        g = g.dropna(subset=["Date","Home","Away","Home_Score","Away_Score"])
        g["game_key"] = g["Date"].dt.strftime("%Y-%m-%d") + "|" + g["Away"].astype(str) + "@" + g["Home"].astype(str)
        if "gamePk" in g.columns:
            pk = pd.to_numeric(g["gamePk"], errors="coerce")
            g.loc[pk.notna(), "game_key"] = "pk:" + pk[pk.notna()].astype("int64").astype(str)
        return g.sort_values(["Date","game_key"]).drop_duplicates("game_key", keep="last").reset_index(drop=True)

    @staticmethod
    def _games_fingerprint(games: pd.DataFrame) -> str:
        if games is None or games.empty: return "empty"
        cols = [c for c in ("game_key","Date","Season","Home","Away","Home_Score","Away_Score") if c in games.columns]
        x = games[cols].copy().sort_values([c for c in ("Date","game_key") if c in cols]).reset_index(drop=True)
        if "Date" in x.columns: x["Date"] = pd.to_datetime(x["Date"], errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S")
        return hashlib.sha256(x.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()

    def _ensure_tracking_tables(self, con):
        con.execute("""CREATE TABLE IF NOT EXISTS predictions (
            prediction_id VARCHAR PRIMARY KEY, created_at TIMESTAMP, game_key VARCHAR, game_date TIMESTAMP,
            home VARCHAR, away VARCHAR, market VARCHAR, selection VARCHAR, prob_ml DOUBLE, prob_mc DOUBLE,
            probability DOUBLE, odds DOUBLE, edge_pp DOUBLE, ev_pct DOUBLE, kelly_pct DOUBLE, accepted BOOLEAN,
            model_version VARCHAR, feature_version VARCHAR, payload_json VARCHAR, settled BOOLEAN DEFAULT FALSE,
            result VARCHAR, profit_units DOUBLE, settled_at TIMESTAMP)""")
        con.execute("CREATE INDEX IF NOT EXISTS idx_predictions_game ON predictions(game_key)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(game_date)")

    def ingest_games(self, df_games: pd.DataFrame) -> int:
        g = self._normalize_games(df_games)
        if g.empty: return 0
        con = self.connect()
        try:
            con.register("incoming_games", g)
            con.execute("CREATE TABLE IF NOT EXISTS games AS SELECT * FROM incoming_games WHERE 1=0")
            existing = {r[1] for r in con.execute("PRAGMA table_info('games')").fetchall()}
            for c in g.columns:
                if c not in existing: con.execute(f'ALTER TABLE games ADD COLUMN "{c}" VARCHAR')
            con.execute("DELETE FROM games USING incoming_games WHERE games.game_key = incoming_games.game_key")
            current = {r[1] for r in con.execute("PRAGMA table_info('games')").fetchall()}
            common = [c for c in g.columns if c in current]; cols = ",".join(f'"{c}"' for c in common)
            con.execute(f"INSERT INTO games ({cols}) SELECT {cols} FROM incoming_games")
            con.execute("CREATE INDEX IF NOT EXISTS idx_games_date ON games(Date)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_games_key ON games(game_key)")
            return len(g)
        finally: con.close()

    def replace_games(self, df_games: pd.DataFrame) -> int:
        """Exact mirror of the canonical source, including deletions/corrections."""
        g = self._normalize_games(df_games)
        if g.empty: return 0
        con = self.connect()
        try:
            try: con.execute("DELETE FROM games")
            except Exception: pass
        finally: con.close()
        return self.ingest_games(g)

    def ensure_fresh_from_repository(self, games_path: str | Path = "data/mlb_games.csv") -> dict:
        path = Path(games_path)
        if not path.exists(): return {"fresh": False, "rebuilt": False, "reason": "source_missing"}
        source = self._normalize_games(pd.read_csv(path)); source_fp = self._games_fingerprint(source)
        con = self.connect()
        try:
            try: stored = con.execute("SELECT * FROM games ORDER BY Date, game_key").df()
            except Exception: stored = pd.DataFrame()
        finally: con.close()
        stored_fp = self._games_fingerprint(self._normalize_games(stored)) if not stored.empty else "empty"
        if source_fp == stored_fp and self.paths.features.exists():
            return {"fresh": True, "rebuilt": False, "games": len(source), "fingerprint": source_fp}
        ingested = self.replace_games(source); rebuilt = self.rebuild_feature_store()
        return {"fresh": True, "rebuilt": True, "games": len(source), "ingested": ingested,
                "features": rebuilt, "fingerprint": source_fp}

    @staticmethod
    def build_leak_safe_features(df_games: pd.DataFrame) -> pd.DataFrame:
        games = MLBDataWarehouse._normalize_games(df_games)
        if games.empty: return pd.DataFrame()
        hist: Dict[str,list] = {}; hh: Dict[Tuple[str,str],list] = {}; last_game: Dict[str,pd.Timestamp] = {}; rows=[]
        for r in games.itertuples(index=False):
            loc, vis, date = str(r.Home), str(r.Away), pd.Timestamp(r.Date)
            w5l,rf5l,ra5l,rd5l = team_state(hist,loc,5); w5v,rf5v,ra5v,rd5v = team_state(hist,vis,5)
            w20l,_,_,rd20l = team_state(hist,loc,20); w20v,_,_,rd20v = team_state(hist,vis,20)
            hwin,hrd,hn = h2h_state(hh,loc,vis,12)
            def rest(team):
                if team not in last_game: return 3.0
                return float(np.clip((date.normalize()-last_game[team].normalize()).days-1,0,7))
            rows.append({"game_key":r.game_key,"Date":date,"Season":int(r.Season),"Home":loc,"Away":vis,
                "home_win5":w5l,"away_win5":w5v,"home_win20":w20l,"away_win20":w20v,
                "home_rf5":rf5l,"away_rf5":rf5v,"home_ra5":ra5l,"away_ra5":ra5v,
                "home_rd5":rd5l,"away_rd5":rd5v,"home_rd20":rd20l,"away_rd20":rd20v,
                "h2h_home_win":hwin,"h2h_home_rd":hrd,"h2h_sample":min(hn,12)/12.0,
                "home_rest_days":rest(loc),"away_rest_days":rest(vis),
                "target_home_win":int(float(r.Home_Score)>float(r.Away_Score)),
                "target_total_runs":float(r.Home_Score)+float(r.Away_Score),
                "target_run_diff":float(r.Home_Score)-float(r.Away_Score),"feature_version":FEATURE_VERSION})
            append_game(hist,hh,loc,vis,float(r.Home_Score),float(r.Away_Score)); last_game[loc]=date; last_game[vis]=date
        return pd.DataFrame(rows)

    def rebuild_feature_store(self) -> int:
        con=self.connect()
        try: games=con.execute("SELECT * FROM games ORDER BY Date, game_key").df()
        finally: con.close()
        features=self.build_leak_safe_features(games)
        if features.empty: return 0
        features.to_parquet(self.paths.features,index=False)
        con=self.connect()
        try:
            con.register("feature_frame",features); con.execute("DROP TABLE IF EXISTS pregame_features")
            con.execute("CREATE TABLE pregame_features AS SELECT * FROM feature_frame")
            con.execute("CREATE INDEX IF NOT EXISTS idx_features_date ON pregame_features(Date)")
        finally: con.close()
        return len(features)

    def training_frame(self, cutoff_date: Optional[str|pd.Timestamp]=None) -> pd.DataFrame:
        con=self.connect()
        try:
            if cutoff_date is None: return con.execute("SELECT * FROM pregame_features ORDER BY Date, game_key").df()
            return con.execute("SELECT * FROM pregame_features WHERE Date < ? ORDER BY Date, game_key",[pd.Timestamp(cutoff_date)]).df()
        finally: con.close()

    def legacy_ml_training_frame(self, df_batting: pd.DataFrame, df_pitching: pd.DataFrame, bat_scale=100.0, pit_scale=4.10) -> pd.DataFrame:
        self.ensure_fresh_from_repository(); f=self.training_frame().copy()
        if f.empty: return f
        bcol=batting_metric(df_batting); pcol=pitching_metric(df_pitching)
        if not bcol or not pcol: return pd.DataFrame()
        b=df_batting[["Team","Season",bcol]].copy(); p=df_pitching[["Team","Season",pcol]].copy()
        b["Team"]=b["Team"].map(normalize_team); p["Team"]=p["Team"].map(normalize_team)
        b["Season"]=pd.to_numeric(b["Season"],errors="coerce"); p["Season"]=pd.to_numeric(p["Season"],errors="coerce")
        b[bcol]=pd.to_numeric(b[bcol],errors="coerce"); p[pcol]=pd.to_numeric(p[pcol],errors="coerce")
        bd=b.dropna().set_index(["Team","Season"])[bcol].to_dict(); pdict=p.dropna().set_index(["Team","Season"])[pcol].to_dict()
        off_h=[]; off_a=[]; pit_h=[]; pit_a=[]
        for r in f.itertuples(index=False):
            sy=int(r.Season)-1
            off_h.append(float(bd.get((r.Home,sy),bat_scale))/max(float(bat_scale),1e-6))
            off_a.append(float(bd.get((r.Away,sy),bat_scale))/max(float(bat_scale),1e-6))
            pit_h.append(float(pdict.get((r.Home,sy),pit_scale))/max(float(pit_scale),1e-6))
            pit_a.append(float(pdict.get((r.Away,sy),pit_scale))/max(float(pit_scale),1e-6))
        f["home_offense"]=off_h; f["away_offense"]=off_a; f["home_pitching"]=pit_h; f["away_pitching"]=pit_a; f["home_field"]=1.0
        return f

    def record_prediction(self, *, game_key, game_date, home, away, market, selection, prob_ml, prob_mc,
                          probability, odds, edge_pp=None, ev_pct=None, kelly_pct=None, accepted=False,
                          model_version="mlb_v6_bigdata", payload=None) -> str:
        pid=str(uuid.uuid4()); con=self.connect()
        try:
            self._ensure_tracking_tables(con)
            con.execute("""INSERT INTO predictions(prediction_id,created_at,game_key,game_date,home,away,market,selection,
                prob_ml,prob_mc,probability,odds,edge_pp,ev_pct,kelly_pct,accepted,model_version,feature_version,payload_json,settled)
                VALUES (?,CURRENT_TIMESTAMP,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,FALSE)""",
                [pid,str(game_key),pd.Timestamp(game_date),normalize_team(home),normalize_team(away),str(market),str(selection),
                 float(prob_ml),float(prob_mc),float(probability),float(odds),None if edge_pp is None else float(edge_pp),
                 None if ev_pct is None else float(ev_pct),None if kelly_pct is None else float(kelly_pct),bool(accepted),
                 str(model_version),FEATURE_VERSION,json.dumps(payload or {},default=str)])
            return pid
        finally: con.close()

    def settle_prediction(self, prediction_id: str, result: str, profit_units: float) -> bool:
        result=str(result).upper().strip()
        if result not in {"WIN","LOSS","PUSH","VOID"}: raise ValueError("result debe ser WIN/LOSS/PUSH/VOID")
        con=self.connect()
        try:
            self._ensure_tracking_tables(con)
            con.execute("UPDATE predictions SET settled=TRUE,result=?,profit_units=?,settled_at=CURRENT_TIMESTAMP WHERE prediction_id=?",
                        [result,float(profit_units),str(prediction_id)])
            return con.execute("SELECT COUNT(*) FROM predictions WHERE prediction_id=? AND settled=TRUE",[str(prediction_id)]).fetchone()[0]==1
        finally: con.close()

    def performance_summary(self, accepted_only=True) -> dict:
        con=self.connect()
        try:
            self._ensure_tracking_tables(con); where="settled=TRUE" + (" AND accepted=TRUE" if accepted_only else "")
            row=con.execute(f"""SELECT COUNT(*), SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END),
                SUM(CASE WHEN result='LOSS' THEN 1 ELSE 0 END), COALESCE(SUM(profit_units),0),
                AVG(CASE WHEN result IN ('WIN','LOSS') THEN CASE WHEN result='WIN' THEN 1.0 ELSE 0.0 END END)
                FROM predictions WHERE {where}""").fetchone()
            n,w,l,profit,hit=row
            return {"settled":int(n or 0),"wins":int(w or 0),"losses":int(l or 0),"profit_units":round(float(profit or 0),4),
                    "hit_rate":None if hit is None else round(float(hit)*100,2)}
        finally: con.close()

    def status(self) -> dict:
        con=self.connect()
        try:
            self._ensure_tracking_tables(con)
            def count(table):
                try: return int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                except Exception: return 0
            return {"backend":"duckdb","db":str(self.paths.db),"feature_version":FEATURE_VERSION,
                    "games":count("games"),"features":count("pregame_features"),"predictions":count("predictions")}
        finally: con.close()

def bootstrap_from_repository(data_dir: str|Path="data", root: str|Path="data/bigdata") -> dict:
    games_path=Path(data_dir)/"mlb_games.csv"
    if not games_path.exists(): raise FileNotFoundError(games_path)
    wh=MLBDataWarehouse(root); games=pd.read_csv(games_path); ingested=wh.replace_games(games); features=wh.rebuild_feature_store()
    out=wh.status(); out.update({"ingested":ingested,"rebuilt_features":features}); return out
