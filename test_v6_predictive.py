import math
import pandas as pd

from modules.advanced_stats import enrich_team_frames
from modules.metric_quality import batting_metric, pitching_metric, row_pitching_value
from modules.ml_mlb import PredictorMLMLB


def test_metric_quality_prefers_real_sources_only_with_coverage():
    legacy_bat = pd.DataFrame({'Team':['NYY','BOS','LAD','SF'], 'Season':[2025]*4, 'OPS_Index':[75,72,78,70], 'wRC+':[75,72,78,70], 'wRC+_Source':['LEGACY_OPS_X100_NOT_REAL_WRCPLUS']*4})
    assert batting_metric(legacy_bat) == 'OPS_Index'
    partial_bat = legacy_bat.copy(); partial_bat.loc[0,'wRC+']=112; partial_bat.loc[0,'wRC+_Source']='FANGRAPHS_REAL_WRCPLUS'
    assert batting_metric(partial_bat) == 'OPS_Index'
    real_bat = legacy_bat.copy(); real_bat['wRC+']=[112,98,109,94]; real_bat['wRC+_Source']='FANGRAPHS_REAL_WRCPLUS'
    assert batting_metric(real_bat) == 'wRC+'

    legacy_pit = pd.DataFrame({'Team':['NYY','BOS','LAD','SF'], 'Season':[2025]*4, 'ERA':[4.0,4.2,3.8,4.1], 'xFIP':[4.0,4.2,3.8,4.1], 'xFIP_Source':['LEGACY_ERA_PROXY_NOT_REAL_XFIP']*4})
    assert pitching_metric(legacy_pit) == 'ERA'
    partial_pit=legacy_pit.copy(); partial_pit.loc[0,'xFIP']=3.7; partial_pit.loc[0,'xFIP_Source']='FANGRAPHS_REAL_XFIP'
    assert pitching_metric(partial_pit) == 'ERA'
    real_pit = legacy_pit.copy(); real_pit['xFIP']=[3.7,4.05,3.55,3.9]; real_pit['xFIP_Source']='FANGRAPHS_REAL_XFIP'
    assert pitching_metric(real_pit) == 'xFIP'
    value, used = row_pitching_value(pd.Series({'ERA':4.2,'FIP':3.9,'xFIP':3.7,'xFIP_Source':'FANGRAPHS_REAL_XFIP'}))
    assert abs(value-3.7) < 1e-9 and used == 'xFIP'


def test_feature_vector_and_model_selection():
    bat=pd.read_csv('data/mlb_batting.csv'); pit=pd.read_csv('data/mlb_pitching.csv'); games=pd.read_csv('data/mlb_games.csv')
    m=PredictorMLMLB()
    f=m._feature_row({}, {}, 'NYY','BOS',100,100,4.1,4.2)
    assert len(f) == 20, len(f)
    assert m.entrenar(bat,pit,games)
    assert m.classifier_family in ('logistic','histgb')
    assert m.runs_family in ('gbr','histgb') and m.diff_family in ('gbr','histgb')
    assert m.validation_brier is not None and math.isfinite(m.validation_brier)
    assert m.validation_runs_mae is not None and math.isfinite(m.validation_runs_mae)
    r=m.predecir_partido('NYY','BOS',100,100,4.1,4.2)
    assert abs(r['Probabilidad_Local'] + r['Probabilidad_Visita'] - 100) < .2
    assert r['Classifier_Family'] == m.classifier_family
    assert r['Runs_Family'] == m.runs_family
    assert r['Diff_Family'] == m.diff_family
    m2=PredictorMLMLB(); assert m2.entrenar(bat,pit,games)
    assert m2.loaded_from_cache is True
    assert (m2.classifier_family,m2.runs_family,m2.diff_family) == (m.classifier_family,m.runs_family,m.diff_family)


def test_advanced_enrichment_is_fail_soft_with_empty_frames():
    a,b,c=enrich_team_frames(pd.DataFrame(),pd.DataFrame(),[2026])
    assert a.empty and b.empty and c.empty


def main():
    tests=[v for k,v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for fn in tests:
        fn(); print('PASS',fn.__name__)
    print(f'V6 predictive tests passed: {len(tests)}')

if __name__=='__main__': main()
