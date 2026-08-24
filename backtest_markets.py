import json, math
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss
from modules.ml_mlb import PredictorMLMLB
from modules.historical_mlb import prepare_games
from modules.team_utils import normalize_team


def cdf(z): return .5*(1+math.erf(z/math.sqrt(2)))

def main():
    bat=pd.read_csv('data/mlb_batting.csv'); pit=pd.read_csv('data/mlb_pitching.csv'); games=prepare_games(pd.read_csv('data/mlb_games.csv'))
    cut=int(len(games)*.85); train=games.iloc[:cut]; test=games.iloc[cut:]
    m=PredictorMLMLB(); assert m.entrenar(bat,pit,train)
    bc='OPS_Index' if 'OPS_Index' in bat else 'wRC+'; pc='ERA' if 'ERA' in pit else 'xFIP'
    b=bat.copy(); p=pit.copy(); b['K']=b.Team.map(normalize_team); p['K']=p.Team.map(normalize_team)
    bd=b.set_index(['K','Season'])[bc].to_dict(); pdic=p.set_index(['K','Season'])[pc].to_dict(); bm=float(pd.to_numeric(b[bc],errors='coerce').median()); pm=float(pd.to_numeric(p[pc],errors='coerce').median())
    y=[]; prob=[]; rl15=[]; rl15y=[]; tot85=[]; tot85y=[]
    for _,r in test.iterrows():
        h,a=normalize_team(r.Home),normalize_team(r.Away); sy=int(r.Season)-1
        pr=m.predecir_partido(h,a,bd.get((h,sy),bm),bd.get((a,sy),bm),pdic.get((h,sy),pm),pdic.get((a,sy),pm))
        hs,as_=float(r.Home_Score),float(r.Away_Score); y.append(int(hs>as_)); prob.append(pr['Probabilidad_Local']/100)
        d=pr['Proyeccion_Handicap_Local']; sr=pr['Sigma_Handicap']; rl15.append(cdf((d+1.5)/sr)); rl15y.append(int((hs-as_)+1.5>0))
        tr=pr['Proyeccion_Carreras']; st=pr['Sigma_Carreras']; tot85.append(cdf((tr-8.5)/st)); tot85y.append(int(hs+as_>8.5))
    out={'n_train':len(train),'n_test':len(test),'moneyline_accuracy':round(accuracy_score(y,np.array(prob)>=.5),4),'moneyline_brier':round(brier_score_loss(y,prob),4),'runline_home_plus_1_5_accuracy':round(accuracy_score(rl15y,np.array(rl15)>=.5),4),'runline_home_plus_1_5_brier':round(brier_score_loss(rl15y,rl15),4),'over_8_5_accuracy':round(accuracy_score(tot85y,np.array(tot85)>=.5),4),'over_8_5_brier':round(brier_score_loss(tot85y,tot85),4),'sigma_runs':round(m.sigma_runs,3),'sigma_diff':round(m.sigma_diff,3)}
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
