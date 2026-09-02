"""Add leak-safe starter Day/Night split features to the isolated research Parquet.

For each target game, only completed starts on STRICTLY PRIOR calendar dates are
used. The matching Day/Night split is shrunk toward the pitcher's overall prior
performance so small split samples do not create extreme noisy signals.
Production does not import this module.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json

import numpy as np
import pandas as pd

from modules.experimental_parquet import OUT, REPORT

STARTER_PERF = Path("data/mlb_starter_performance_history.csv")
PRIOR_IP = 25.0
MIN_SPLIT_IP = 5.0
METRICS = ("era", "whip", "k_pct", "bb_pct", "kbb_pct", "hr9")


def _num(v):
    return pd.to_numeric(v, errors="coerce")


def _agg(rows):
    if not rows:
        return None
    g = pd.DataFrame(rows)
    sums = {c: float(_num(g[c]).fillna(0).sum()) for c in ("IP", "ER", "H", "BB", "SO", "HR", "BF")}
    ip, bf = sums["IP"], sums["BF"]
    if ip <= 0 or bf <= 0:
        return None
    k = 100.0 * sums["SO"] / bf
    bb = 100.0 * sums["BB"] / bf
    return {"ip": ip, "era": 9.0*sums["ER"]/ip, "whip": (sums["H"]+sums["BB"])/ip,
            "k_pct": k, "bb_pct": bb, "kbb_pct": k-bb, "hr9": 9.0*sums["HR"]/ip}


def _shrink(split, overall):
    if not split or not overall or split["ip"] < MIN_SPLIT_IP:
        return None
    w = split["ip"] / (split["ip"] + PRIOR_IP)
    out = {"split_ip": split["ip"], "weight": w}
    for m in METRICS:
        out[m] = w*split[m] + (1.0-w)*overall[m]
        out[f"delta_{m}"] = out[m] - overall[m]
    return out


def main():
    if not OUT.exists() or not STARTER_PERF.exists():
        raise RuntimeError("Missing research Parquet or starter performance history")
    frame = pd.read_parquet(OUT)
    hist = pd.read_csv(STARTER_PERF, low_memory=False)
    need = {"GameID","Date","PitcherID","DayNight","IP","ER","H","BB","SO","HR","BF"}
    if not need.issubset(hist.columns):
        raise RuntimeError(f"Starter history lacks Day/Night fields: {sorted(need-set(hist.columns))}")
    hist["Date"] = pd.to_datetime(hist["Date"], errors="coerce").dt.normalize()
    hist["DayNight"] = hist["DayNight"].astype(str).str.lower().where(lambda s: s.isin(["day","night"]))
    for c in ["GameID","PitcherID","IP","ER","H","BB","SO","HR","BF"]:
        hist[c] = _num(hist[c])
    hist = hist.dropna(subset=["Date","PitcherID"]).sort_values(["Date","GameID"])

    frame["_date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
    if "day_game" in frame.columns:
        dg = _num(frame["day_game"])
        frame["_dn"] = np.where(dg.eq(1), "day", np.where(dg.eq(0), "night", None))
    else:
        frame["_dn"] = None
    # Recover target-game Day/Night directly from the official starter log when
    # canonical historical game context did not already carry day_game.
    game_dn = hist.dropna(subset=["GameID","DayNight"]).drop_duplicates("GameID").set_index("GameID")["DayNight"].to_dict()
    missing = frame["_dn"].isna()
    frame.loc[missing, "_dn"] = [game_dn.get(int(x)) if pd.notna(x) else None for x in frame.loc[missing, "gamePk"]]
    frame["day_game"] = np.where(frame["_dn"].eq("day"), 1.0, np.where(frame["_dn"].eq("night"), 0.0, np.nan))

    by_pid = defaultdict(list)
    for r in hist.itertuples(index=False):
        by_pid[int(r.PitcherID)].append({"Date":pd.Timestamp(r.Date),"GameID":int(r.GameID),"DayNight":r.DayNight,
            "IP":r.IP,"ER":r.ER,"H":r.H,"BB":r.BB,"SO":r.SO,"HR":r.HR,"BF":r.BF})

    feature_names = []
    for side in ("home","away"):
        for m in METRICS:
            feature_names += [f"{side}_starter_dn_{m}", f"{side}_starter_dn_delta_{m}"]
        feature_names += [f"{side}_starter_dn_ip", f"{side}_starter_dn_weight"]
    vals = {c:{} for c in feature_names}

    order = frame.sort_values(["_date","gamePk"], kind="mergesort").index
    for idx in order:
        d, condition = frame.at[idx,"_date"], frame.at[idx,"_dn"]
        if pd.isna(d) or condition not in {"day","night"}:
            continue
        for side in ("home","away"):
            idcol=f"{side}_starter_id"
            if idcol not in frame.columns or pd.isna(frame.at[idx,idcol]):
                continue
            pid=int(float(frame.at[idx,idcol])); prior=[r for r in by_pid.get(pid,[]) if r["Date"] < d]
            split=[r for r in prior if r.get("DayNight") == condition]
            shrunk=_shrink(_agg(split), _agg(prior))
            if not shrunk:
                continue
            vals[f"{side}_starter_dn_ip"][idx]=shrunk["split_ip"]
            vals[f"{side}_starter_dn_weight"][idx]=shrunk["weight"]
            for m in METRICS:
                vals[f"{side}_starter_dn_{m}"][idx]=shrunk[m]
                vals[f"{side}_starter_dn_delta_{m}"][idx]=shrunk[f"delta_{m}"]

    for c,mapping in vals.items():
        frame[c]=pd.Series(mapping,dtype=float)
    frame=frame.drop(columns=["_date","_dn"],errors="ignore")
    frame.to_parquet(OUT,index=False)

    coverage={c:round(float(frame[c].notna().mean()),4) for c in feature_names}
    payload=json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {}
    payload["starter_day_night"]={"source":str(STARTER_PERF),"strictly_prior_dates":True,"prior_ip":PRIOR_IP,
        "min_split_ip":MIN_SPLIT_IP,"metrics":list(METRICS),"coverage":coverage,
        "note":"matching Day/Night split shrunk toward pitcher overall prior performance"}
    REPORT.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps({"rows":len(frame),"day_game_coverage":round(float(frame["day_game"].notna().mean()),4),
        "home_dn_era_coverage":coverage.get("home_starter_dn_era",0),"away_dn_era_coverage":coverage.get("away_starter_dn_era",0)},indent=2))


if __name__ == "__main__":
    main()
