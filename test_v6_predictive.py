import math
import pandas as pd

from modules.advanced_stats import enrich_team_frames  # import must be network-free
from modules.metric_quality import batting_metric, pitching_metric, row_pitching_value
from modules.ml_mlb import PredictorMLMLB


def test_metric_quality_prefers_real_sources():
    legacy_bat = pd.DataFrame({'Team':['NYY','BOS'], 'Season':[2025,2025], 'OPS_Index':[75,72], 'wRC+':[75,72], 'wRC+_Source':['LEGACY_OPS_X100_NOT_REAL_WRCPLUS']*2})
    assert batting_metric(legacy_bat) == 'OPS_Index'
    real_bat = legacy_bat.copy(); real_bat['wRC+']=[112,98]; real_bat['wRC+_Source']='FANGRAPHS_REAL_WRCPLUS'
    assert batting_metric(real_bat) == 'wRC+'

    legacy_pit = pd.DataFrame({'Team':['NYY','BOS'], 'Season':[2025,2025], 'ERA':[4.0,4.2], 'xFIP':[4.0,4.2], 'xFIP_Source':['LEGACY_ERA_PROXY_NOT_REAL_XFIP']*2})
    assert pitching_metric(legacy_pit) == 'ERA'
    real_pit = legacy_pit.copy(); real_pit['xFIP']=[3.7,4.05]; real_pit['xFIP_Source']='FANGRAPHS_REAL_XFIP'
    assert pitching_metric(real_pit) == 'xFIP'
    value, used = row_pitching_value(pd.Series({'ERA':4.2,'FIP':3.9,'xFIP':3.7,'xFIP_Source':'FANGRAPHS_REAL_XFIP'}))
    assert abs(value-3.7) < 1e-9 and used == 'xFIP'


def test_feature_vector_and_model_selection():
    bat=pd.read_csv('data/mlb_batting.csv'); pit=pd.read_csv('data/mlb_pitching.csv'); games=pd.read_csv('data/mlb_games.csv')
    m=PredictorMLMLB()
    empty={}
    f=m._feature_row(empty, empty, 'NYY','BOS',100,100,4.1,4.2)
    assert len(f) == 28, len(f)
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
