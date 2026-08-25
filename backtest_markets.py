import json, math
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss

from modules.historical_mlb import prepare_games
from modules.ml_mlb import PredictorMLMLB
from modules.team_utils import normalize_team


def cdf(z):
    return .5 * (1 + math.erf(z / math.sqrt(2)))


def selective(y, probs, threshold):
    y = np.asarray(y, dtype=int); p = np.asarray(probs, dtype=float); mask = p >= threshold
    n = int(mask.sum())
    if not n:
        return {'n': 0, 'hit_rate': None, 'avg_prob': None, 'calibration_error_pp': None}
    hit = float(y[mask].mean()); avg = float(p[mask].mean())
    return {'n': n, 'hit_rate': round(hit, 4), 'avg_prob': round(avg, 4),
            'calibration_error_pp': round((avg-hit)*100, 2)}


def main():
    bat = pd.read_csv('data/mlb_batting.csv')
    pit = pd.read_csv('data/mlb_pitching.csv')
    games = prepare_games(pd.read_csv('data/mlb_games.csv'))
    cut = int(len(games) * .85); train = games.iloc[:cut]; test = games.iloc[cut:]
    m = PredictorMLMLB(); assert m.entrenar(bat, pit, train)

    bc = 'OPS_Index' if 'OPS_Index' in bat.columns else 'wRC+'
    pc = 'ERA' if 'ERA' in pit.columns else 'xFIP'
    b = bat.copy(); p = pit.copy(); b['K'] = b.Team.map(normalize_team); p['K'] = p.Team.map(normalize_team)
    b['Season'] = pd.to_numeric(b['Season'], errors='coerce'); p['Season'] = pd.to_numeric(p['Season'], errors='coerce')
    b[bc] = pd.to_numeric(b[bc], errors='coerce'); p[pc] = pd.to_numeric(p[pc], errors='coerce')
    bd = b.dropna(subset=['K','Season',bc]).set_index(['K','Season'])[bc].to_dict()
    pdic = p.dropna(subset=['K','Season',pc]).set_index(['K','Season'])[pc].to_dict()
    bm = float(b[bc].median()); pm = float(p[pc].median())

    y=[]; prob=[]; rl15=[]; rl15y=[]; over=[]; overy=[]; under=[]; undery=[]
    for _, r in test.iterrows():
        h, a = normalize_team(r.Home), normalize_team(r.Away); sy = int(r.Season)-1
        pr = m.predecir_partido(h, a, bd.get((h,sy),bm), bd.get((a,sy),bm), pdic.get((h,sy),pm), pdic.get((a,sy),pm))
        hs, as_ = float(r.Home_Score), float(r.Away_Score)
        y.append(int(hs>as_)); prob.append(pr['Probabilidad_Local']/100)
        d = pr['Proyeccion_Handicap_Local']; sr = pr['Sigma_Handicap']
        rl15.append(cdf((d+1.5)/sr)); rl15y.append(int((hs-as_)+1.5>0))
        tr = pr['Proyeccion_Carreras']; st = pr['Sigma_Carreras']
        po = cdf((tr-8.5)/st); pu = cdf((8.5-tr)/st)
        over.append(po); under.append(pu); overy.append(int(hs+as_>8.5)); undery.append(int(hs+as_<8.5))

    rl_base=float(np.mean(rl15y)); over_base=float(np.mean(overy)); under_base=float(np.mean(undery)); ml_base=float(np.mean(y))
    out={
      'n_train':len(train),'n_test':len(test),
      'moneyline_accuracy':round(accuracy_score(y,np.array(prob)>=.5),4),'moneyline_brier':round(brier_score_loss(y,prob),4),'moneyline_home_base_rate':round(ml_base,4),
      'runline_home_plus_1_5_accuracy':round(accuracy_score(rl15y,np.array(rl15)>=.5),4),'runline_home_plus_1_5_brier':round(brier_score_loss(rl15y,rl15),4),
      'runline_home_plus_1_5_base_rate':round(rl_base,4),'runline_base_brier':round(brier_score_loss(rl15y,[rl_base]*len(rl15y)),4),
      'runline_signals_ge_54':selective(rl15y,rl15,.54),'runline_signals_ge_56':selective(rl15y,rl15,.56),'runline_signals_ge_60':selective(rl15y,rl15,.60),
      'over_8_5_accuracy':round(accuracy_score(overy,np.array(over)>=.5),4),'over_8_5_brier':round(brier_score_loss(overy,over),4),'over_8_5_base_rate':round(over_base,4),
      'under_8_5_accuracy':round(accuracy_score(undery,np.array(under)>=.5),4),'under_8_5_brier':round(brier_score_loss(undery,under),4),'under_8_5_base_rate':round(under_base,4),
      'over_signals_ge_52':selective(overy,over,.52),'over_signals_ge_54':selective(overy,over,.54),'over_signals_ge_56':selective(overy,over,.56),
      'under_signals_ge_52':selective(undery,under,.52),'under_signals_ge_54':selective(undery,under,.54),'under_signals_ge_56':selective(undery,under,.56),
      'sigma_runs':round(m.sigma_runs,3),'sigma_diff':round(m.sigma_diff,3),
      'historical_odds_available':False,'roi_claim_allowed':False
    }
    print(json.dumps(out,indent=2))

if __name__=='__main__':
    main()
