import inspect
import numpy as np
import pandas as pd

from modules.signal_features import MODEL_ADVANCED_COLUMNS, build_advanced_signal_frame, build_live_signal_row, coverage_report
from modules.ml_mlb import PredictorMLMLB
from modules.montecarlo_mlb import _team_schedule_load


def _frames():
    batting=pd.DataFrame([
        {'Team':'NYY','Season':2025,'wOBA':.340,'ISO':.190,'BB%':9.5,'K%':20.0,'EV':90.5,'HardHit%':44.0,'Barrel%':9.5,'OPS_vs_L':.790,'OPS_vs_R':.760,'OBP_vs_L':.345,'OBP_vs_R':.330,'SLG_vs_L':.445,'SLG_vs_R':.430},
        {'Team':'BOS','Season':2025,'wOBA':.315,'ISO':.165,'BB%':8.0,'K%':24.0,'EV':88.0,'HardHit%':39.0,'Barrel%':7.0,'OPS_vs_L':.710,'OPS_vs_R':.700,'OBP_vs_L':.315,'OBP_vs_R':.310,'SLG_vs_L':.395,'SLG_vs_R':.390},
        {'Team':'LAD','Season':2025,'wOBA':.330,'ISO':.180,'BB%':9.0,'K%':21.0,'EV':89.5,'HardHit%':42.0,'Barrel%':8.5,'OPS_vs_L':.760,'OPS_vs_R':.750,'OBP_vs_L':.335,'OBP_vs_R':.330,'SLG_vs_L':.425,'SLG_vs_R':.420},
    ])
    pitching=pd.DataFrame([
        {'Team':'NYY','Season':2025,'FIP':3.70,'xFIP':3.80,'SIERA':3.75,'WHIP':1.18,'K-BB%':16.0,'GB%':44.0,'HR/9':1.05},
        {'Team':'BOS','Season':2025,'FIP':4.25,'xFIP':4.15,'SIERA':4.10,'WHIP':1.31,'K-BB%':12.0,'GB%':40.0,'HR/9':1.25},
        {'Team':'LAD','Season':2025,'FIP':3.95,'xFIP':3.90,'SIERA':3.88,'WHIP':1.22,'K-BB%':15.0,'GB%':43.0,'HR/9':1.10},
    ])
    return batting,pitching


def test_training_and_live_signal_vectors_match_exactly():
    bat,pit=_frames();frame=pd.DataFrame([{'Home':'NYY','Away':'BOS','Season':2026,'home_rest_days':1,'away_rest_days':2}])
    train=build_advanced_signal_frame(frame,bat,pit).iloc[0].to_numpy(float);live=build_live_signal_row('NYY','BOS',2026,bat,pit,1,2)
    assert len(train)==len(MODEL_ADVANCED_COLUMNS)==len(live);np.testing.assert_allclose(train,live,rtol=0,atol=1e-12)


def test_signal_direction_is_semantically_correct():
    bat,pit=_frames();d=build_advanced_signal_frame(pd.DataFrame([{'Home':'NYY','Away':'BOS','Season':2026}]),bat,pit).iloc[0]
    assert d.home_woba_rel>d.away_woba_rel;assert d.home_k_rel>d.away_k_rel;assert d.home_fip_rel>d.away_fip_rel;assert d.home_kbb_rel>d.away_kbb_rel;assert d.home_ops_l_rel>d.away_ops_l_rel;assert d.home_ops_r_rel>d.away_ops_r_rel


def test_missing_signal_is_neutral_not_fabricated():
    bat,pit=_frames();bat=bat.drop(columns=['Barrel%','OPS_vs_L']);pit=pit.drop(columns=['SIERA']);d=build_advanced_signal_frame(pd.DataFrame([{'Home':'NYY','Away':'BOS','Season':2026}]),bat,pit).iloc[0]
    assert d.home_barrel_rel==1.0 and d.away_barrel_rel==1.0;assert d.home_siera_rel==1.0 and d.away_siera_rel==1.0;assert d.home_ops_l_rel==1.0 and d.away_ops_l_rel==1.0


def test_coverage_is_explicit():
    bat,pit=_frames();report=coverage_report(bat,pit);assert report['batting:wOBA']==1.0;assert report['batting:OPS_vs_L']==1.0;assert report['pitching:xFIP']==1.0;assert report['context:rest_days']==1.0
    bat.loc[0,'EV']=np.nan;report=coverage_report(bat,pit);assert 0.6<report['batting:EV']<0.7


def test_advanced_model_requires_real_validation_gain():
    base={'brier':.247,'runs_mae':3.57,'diff_mae':3.55};worse={'brier':.248,'runs_mae':3.55,'diff_mae':3.54};better={'brier':.244,'runs_mae':3.56,'diff_mae':3.54}
    assert PredictorMLMLB._advanced_wins(base,worse)[0] is False;assert PredictorMLMLB._advanced_wins(base,better)[0] is True


def test_date_aware_contract_is_explicit():
    pred=inspect.signature(PredictorMLMLB.predecir_partido);update=inspect.signature(PredictorMLMLB.actualizar_resultado)
    assert 'game_date' in pred.parameters;assert 'game_date' in update.parameters


def test_bullpen_schedule_load_is_small_and_directional():
    rows=[];base=pd.Timestamp('2026-08-20')
    for i,day in enumerate([0,1,2]):rows.append({'GameID':9000+i,'Date':base+pd.Timedelta(days=day),'Season':2026,'GameType':'R','Away':'BOS','Home':'NYY','Away_Score':3,'Home_Score':4})
    factor,rest,recent,source=_team_schedule_load(pd.DataFrame(rows),'NYY');assert source=='completed_schedule_proxy';assert 0.975<=factor<=1.035;assert rest==0;assert recent>=2;assert factor>1.0


if __name__=='__main__':
    tests=[v for k,v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for t in tests:t();print('PASS',t.__name__)
    print('Signal feature tests passed:',len(tests))
