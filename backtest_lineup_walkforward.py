"""Nested walk-forward validation of confirmed MLB starting-lineup strength.

The candidate never sees postgame or same-day batting results. Each hitter profile is
built only from official boxscores strictly before the game date, using a rolling
365-day window. Games with insufficient hitter history fall back to the production
baseline and are excluded from the promotion sample rather than imputing fake stats.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from modules.historical_mlb import prepare_games
from modules.metric_quality import batting_metric, pitching_metric
from modules.ml_mlb import PredictorMLMLB
from modules.team_utils import normalize_team

OUT = Path("artifacts/lineup_walkforward_predictions.csv")
REPORT = Path("artifacts/lineup_walkforward_report.json")
C_GRID = (0.005, 0.01, 0.03, 0.10)
MIN_PLAYER_PA = 40.0
MIN_QUALIFIED = 7
FEATURE_SETS = {
    "lineup_ops": ("lu_ops_diff",),
    "lineup_onbase_power": ("lu_obp_diff", "lu_slg_diff"),
    "lineup_discipline": ("lu_bb_rate_diff", "lu_k_rate_diff"),
    "lineup_core": ("lu_ops_diff", "lu_bb_rate_diff", "lu_k_rate_diff"),
    "lineup_full": ("lu_obp_diff", "lu_slg_diff", "lu_bb_rate_diff", "lu_k_rate_diff"),
}
STAT_COLS = ("AB", "H", "2B", "3B", "HR", "BB", "SO", "HBP", "SF")


def _date_boundary(games, frac=.70):
    d = pd.Series(games.Date.dt.normalize().dropna().unique()).sort_values().reset_index(drop=True)
    return pd.Timestamp(d.iloc[max(1, min(len(d)-1, int(len(d)*frac)))])


def _maps(df, metric):
    x = df.copy(); x["Team"] = x.Team.map(normalize_team)
    x["Season"] = pd.to_numeric(x.Season, errors="coerce")
    x[metric] = pd.to_numeric(x[metric], errors="coerce")
    return x.dropna(subset=["Team", "Season", metric]).set_index(["Team", "Season"])[metric].to_dict()


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-5, 1-1e-5)
    return np.log(p/(1-p))


def _design(df, cols):
    return np.hstack([_logit(df.baseline_home_prob).reshape(-1, 1), df[list(cols)].to_numpy(float)])


def _brier(y, p): return float(np.mean((np.asarray(p)-np.asarray(y))**2))
def _ll(y, p):
    p = np.clip(np.asarray(p), 1e-9, 1-1e-9); y = np.asarray(y)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))


def _ece(y, p):
    y=np.asarray(y); p=np.asarray(p); total=0.; edges=np.linspace(0,1,11)
    for i in range(10):
        m=(p>=edges[i])&((p<edges[i+1]) if i<9 else (p<=edges[i+1]))
        if m.any(): total += (m.sum()/len(y))*abs(float(p[m].mean()-y[m].mean()))
    return float(total)


def _fit(df, cols, c):
    model=make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, solver="lbfgs", random_state=42))
    model.fit(_design(df, cols), df.actual_home_win.astype(int))
    return model


def _prepare_hitter_history(path):
    h=pd.read_csv(path, low_memory=False); h["Date"]=pd.to_datetime(h.Date, errors="coerce")
    h["PlayerID"]=pd.to_numeric(h.PlayerID, errors="coerce")
    for c in STAT_COLS: h[c]=pd.to_numeric(h.get(c), errors="coerce").fillna(0.0)
    return h.dropna(subset=["Date","PlayerID"]).sort_values(["PlayerID","Date","GameID"])


class PlayerRolling365:
    def __init__(self, history):
        self.data={}
        for pid,g in history.groupby(history.PlayerID.astype(int), sort=False):
            dates=g.Date.dt.normalize().to_numpy(dtype="datetime64[ns]")
            vals=g[list(STAT_COLS)].to_numpy(float)
            cs=np.vstack([np.zeros((1,len(STAT_COLS))), np.cumsum(vals,axis=0)])
            self.data[int(pid)]=(dates,cs)

    def profile(self, pid, date):
        z=self.data.get(int(pid))
        if z is None: return None
        dates,cs=z; d=np.datetime64(pd.Timestamp(date).normalize())
        lo=np.searchsorted(dates, d-np.timedelta64(365,"D"), side="left")
        hi=np.searchsorted(dates, d, side="left")
        if hi<=lo: return None
        v=cs[hi]-cs[lo]; s=dict(zip(STAT_COLS,v))
        pa=s["AB"]+s["BB"]+s["HBP"]+s["SF"]
        if pa < MIN_PLAYER_PA or s["AB"] <= 0: return None
        tb=s["H"]+s["2B"]+2*s["3B"]+3*s["HR"]
        obp=(s["H"]+s["BB"]+s["HBP"])/max(pa,1.0)
        slg=tb/max(s["AB"],1.0)
        return {"pa":pa,"obp":obp,"slg":slg,"ops":obp+slg,"bb_rate":s["BB"]/pa,"k_rate":s["SO"]/pa}


def _lineup_features(lineups, roller, games):
    lu=lineups.copy(); lu["GameID"]=pd.to_numeric(lu.GameID, errors="coerce"); lu["PlayerID"]=pd.to_numeric(lu.PlayerID, errors="coerce")
    grouped={(int(gid),side):x.sort_values("BattingOrder") for (gid,side),x in lu.dropna(subset=["GameID","PlayerID"]).groupby(["GameID","Side"])}
    rows=[]
    for _,r in games.iterrows():
        gid=int(r.GameID) if pd.notna(r.get("GameID")) else int(r.gamePk)
        sides={}
        for side in ("home","away"):
            x=grouped.get((gid,side)); prof=[]
            if x is not None:
                for pid in x.PlayerID.astype(int).head(9):
                    p=roller.profile(pid,r.Date)
                    if p is not None: prof.append(p)
            q=len(prof)
            if q:
                sides[side]={k:float(np.mean([p[k] for p in prof])) for k in ("ops","obp","slg","bb_rate","k_rate")}
            else:
                sides[side]={k:0.0 for k in ("ops","obp","slg","bb_rate","k_rate")}
            sides[side]["qualified"]=q
        z={f"lu_{k}_diff":sides["home"][k]-sides["away"][k] for k in ("ops","obp","slg","bb_rate","k_rate")}
        z["home_qualified"]=sides["home"]["qualified"]; z["away_qualified"]=sides["away"]["qualified"]
        z["lineup_valid"]=float(sides["home"]["qualified"]>=MIN_QUALIFIED and sides["away"]["qualified"]>=MIN_QUALIFIED)
        rows.append(z)
    return pd.DataFrame(rows,index=games.index)


def _select(train):
    days=pd.Series(train.Date.dt.normalize().unique()).sort_values().reset_index(drop=True)
    cut=pd.Timestamp(days.iloc[max(1,min(len(days)-1,int(len(days)*.75)))])
    tr=train[train.Date.dt.normalize()<cut]; va=train[train.Date.dt.normalize()>=cut]
    y=va.actual_home_win.astype(int); base=va.baseline_home_prob
    bb,be=_brier(y,base),_ece(y,base); trials=[]
    for name,cols in FEATURE_SETS.items():
        for c in C_GRID:
            m=_fit(tr,cols,c); p=m.predict_proba(_design(va,cols))[:,1]
            b,l,e=_brier(y,p),_ll(y,p),_ece(y,p)
            score=b+.15*l+.10*max(0,e-be)+.25*max(0,b-bb)
            trials.append({"name":name,"C":c,"features":list(cols),"score":score,"brier":b,"log_loss":l,"ece":e})
    trials.sort(key=lambda x:(x["score"],x["brier"],x["log_loss"],len(x["features"])))
    return trials[0],trials[:5],str(cut.date())


def main():
    bat=pd.read_csv("data/mlb_batting.csv"); pit=pd.read_csv("data/mlb_pitching.csv")
    games=prepare_games(pd.read_csv("data/mlb_games.csv")); boundary=_date_boundary(games)
    train=games[games.Date.dt.normalize()<boundary]; later=games[games.Date.dt.normalize()>=boundary]
    model=PredictorMLMLB(); assert model.entrenar(bat,pit,train)
    bc,pc=batting_metric(bat),pitching_metric(pit); bd,pdct=_maps(bat,bc),_maps(pit,pc)
    bmed=float(pd.to_numeric(bat[bc],errors="coerce").median()); pmed=float(pd.to_numeric(pit[pc],errors="coerce").median())
    rows=[]
    for day,dg in later.groupby(later.Date.dt.normalize(),sort=True):
        pending=[]
        for idx,r in dg.iterrows():
            home,away=normalize_team(r.Home),normalize_team(r.Away); sy=int(r.Season)-1
            pred=model.predecir_partido(home,away,float(bd.get((home,sy),bmed)),float(bd.get((away,sy),bmed)),float(pdct.get((home,sy),pmed)),float(pdct.get((away,sy),pmed)),game_date=r.Date)
            hs,aw=float(r.Home_Score),float(r.Away_Score)
            rows.append({"idx":idx,"Date":r.Date,"GameID":r.get("GameID",r.get("gamePk")),"actual_home_win":int(hs>aw),"actual_total_runs":hs+aw,"baseline_home_prob":float(pred["Probabilidad_Local"])/100.,"baseline_total_runs":float(pred["Proyeccion_Carreras"])})
            pending.append((home,away,hs,aw,r.Date))
        for x in pending: model.actualizar_resultado(x[0],x[1],x[2],x[3],game_date=x[4])
    frame=pd.DataFrame(rows).set_index("idx")
    lineups=pd.read_csv("data/mlb_lineup_history.csv",low_memory=False); hitters=_prepare_hitter_history("data/mlb_hitter_game_history.csv")
    roller=PlayerRolling365(hitters)
    features=_lineup_features(lineups,roller,later.loc[frame.index])
    frame=frame.join(features).sort_values("Date").reset_index(drop=True); frame.Date=pd.to_datetime(frame.Date)
    all_rows=len(frame); valid=frame[frame.lineup_valid>=1].copy(); coverage=len(valid)/max(all_rows,1)
    if len(valid)<900: raise SystemExit(f"Insufficient qualified lineup sample: {len(valid)} rows ({coverage:.1%})")
    days=pd.Series(valid.Date.dt.normalize().unique()).sort_values().reset_index(drop=True)
    rem=days.iloc[max(1,int(len(days)*.45)):].reset_index(drop=True); pooled=[]; reps=[]
    for fi,ch in enumerate(np.array_split(rem,3),1):
        start,end=pd.Timestamp(ch.iloc[0]),pd.Timestamp(ch.iloc[-1]); fit=valid[valid.Date.dt.normalize()<start]; test=valid[(valid.Date.dt.normalize()>=start)&(valid.Date.dt.normalize()<=end)]
        if len(fit)<700 or len(test)<250: continue
        best,top,cut=_select(fit); m=_fit(fit,best["features"],best["C"]); test=test.copy()
        test["candidate_home_prob"]=m.predict_proba(_design(test,best["features"]))[:,1]
        test["candidate_total_runs"]=test.baseline_total_runs; test["walkforward_fold"]=fi
        pooled.append(test); reps.append({"fold":fi,"test_start":str(start.date()),"test_end":str(end.date()),"train_rows":len(fit),"test_rows":len(test),"inner_cut":cut,"selected":best,"top5":top})
    if not pooled: raise SystemExit("No valid walk-forward folds")
    out=pd.concat(pooled).sort_values("Date"); OUT.parent.mkdir(exist_ok=True); out.to_csv(OUT,index=False)
    report={"policy":"nested_walkforward_confirmed_lineup_v1","rows":len(out),"source_rows":all_rows,"qualified_rows":len(valid),"qualified_coverage":coverage,"min_player_pa":MIN_PLAYER_PA,"min_qualified_hitters":MIN_QUALIFIED,"folds":reps}
    REPORT.write_text(json.dumps(report,indent=2)); print("LINEUP_WALKFORWARD",json.dumps(report)); print(f"OK {OUT} rows={len(out)} qualified_coverage={coverage:.1%}")


if __name__=="__main__": main()
