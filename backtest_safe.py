import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error

from modules.historical_mlb import prepare_games
from modules.metric_quality import batting_metric, pitching_metric
from modules.ml_mlb import PredictorMLMLB
from modules.team_utils import normalize_team


def calibration_bins(y, probs):
    y=np.asarray(y,dtype=int); p=np.asarray(probs,dtype=float); bins=[]
    for lo,hi in [(0.35,0.45),(0.45,0.50),(0.50,0.55),(0.55,0.60),(0.60,0.65),(0.65,0.75)]:
        mask=(p>=lo)&(p<hi); n=int(mask.sum())
        if n:
            bins.append({'range':f'{lo:.2f}-{hi:.2f}','n':n,'avg_pred':round(float(p[mask].mean()),4),
                         'actual_home_win':round(float(y[mask].mean()),4),
                         'calibration_error_pp':round(float((p[mask].mean()-y[mask].mean())*100),2)})
    return bins


def selective_precision(y, probs):
    y=np.asarray(y,dtype=int); p=np.asarray(probs,dtype=float)
    picks=(p>=.5).astype(int); confidence=np.maximum(p,1-p); correct=(picks==y)
    out=[]
    for threshold in (.52,.54,.55,.56,.58,.60,.62,.65):
        mask=confidence>=threshold; n=int(mask.sum())
        if not n:
            continue
        out.append({
            'min_confidence':round(threshold,2),
            'n':n,
            'coverage':round(n/len(y),4),
            'hit_rate':round(float(correct[mask].mean()),4),
            'avg_confidence':round(float(confidence[mask].mean()),4),
            'calibration_error_pp':round(float((confidence[mask].mean()-correct[mask].mean())*100),2),
        })
    return out


def main():
    bat=pd.read_csv('data/mlb_batting.csv'); pit=pd.read_csv('data/mlb_pitching.csv'); games=prepare_games(pd.read_csv('data/mlb_games.csv'))
    cut=int(len(games)*0.85); train=games.iloc[:cut].copy(); test=games.iloc[cut:].copy()
    model=PredictorMLMLB()
    if not model.entrenar(bat,pit,train): raise RuntimeError('No se pudo entrenar el modelo')

    bc=batting_metric(bat); pc=pitching_metric(pit)
    if not bc or not pc: raise RuntimeError('No hay métricas válidas')
    b=bat.copy(); p=pit.copy(); b['K']=b['Team'].map(normalize_team); p['K']=p['Team'].map(normalize_team)
    b['Season']=pd.to_numeric(b['Season'],errors='coerce'); p['Season']=pd.to_numeric(p['Season'],errors='coerce')
    b[bc]=pd.to_numeric(b[bc],errors='coerce'); p[pc]=pd.to_numeric(p[pc],errors='coerce')
    bd=b.dropna(subset=['K','Season',bc]).set_index(['K','Season'])[bc].to_dict(); pdict=p.dropna(subset=['K','Season',pc]).set_index(['K','Season'])[pc].to_dict()
    bmed=float(b[bc].median()); pmed=float(p[pc].median())

    y=[]; probs=[]; runs_true=[]; runs_pred=[]
    for _,r in test.iterrows():
        h,a,season=normalize_team(r['Home']),normalize_team(r['Away']),int(r['Season']); sy=season-1
        pred=model.predecir_partido(h,a,float(bd.get((h,sy),bmed)),float(bd.get((a,sy),bmed)),float(pdict.get((h,sy),pmed)),float(pdict.get((a,sy),pmed)))
        hs,as_=float(r['Home_Score']),float(r['Away_Score'])
        probs.append(pred['Probabilidad_Local']/100.0); y.append(int(hs>as_)); runs_true.append(hs+as_); runs_pred.append(float(pred['Proyeccion_Carreras']))
        model.actualizar_resultado(h,a,hs,as_)

    probs=np.asarray(probs,float); y=np.asarray(y,int); picks=(probs>=.5).astype(int); base_p=float(y.mean()); baseline_probs=np.full(len(y),base_p)
    result={
        'n_train':len(train),'n_test':len(test),'batting_metric':bc,'pitching_metric':pc,
        'classifier_family':model.classifier_family,'runs_family':model.runs_family,'diff_family':model.diff_family,
        'accuracy':round(accuracy_score(y,picks),4),'baseline_accuracy_home_rate':round(max(base_p,1-base_p),4),
        'brier':round(brier_score_loss(y,probs),4),'baseline_brier':round(brier_score_loss(y,baseline_probs),4),
        'logloss':round(log_loss(y,probs,labels=[0,1]),4),'baseline_logloss':round(log_loss(y,baseline_probs,labels=[0,1]),4),
        'runs_mae':round(mean_absolute_error(runs_true,runs_pred),3),'calibration_bins':calibration_bins(y,probs),
        'selective_moneyline_precision':selective_precision(y,probs),
        'internal_validation_brier':model.validation_brier,'internal_validation_runs_mae':model.validation_runs_mae,
        'historical_odds_available':False,'roi_claim_allowed':False,
        'note':'Walk-forward real; resultados se incorporan solo después de predecir. Sin cuotas históricas no se afirma ROI.'
    }
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
