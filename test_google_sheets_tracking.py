import os
import sys
from types import SimpleNamespace
from modules.google_sheets_ledger import (
    record_key, sync_rows, SHEET_HEADERS, _sheet_id, _credentials_payload, _schema_action
)


def test_record_key_is_stable_and_market_specific():
    row = {
        'game_date': '2026-08-25', 'game_pk': 123, 'market': 'Moneyline',
        'selection': 'Gana Local (NYY)', 'model_version': 'v6'
    }
    k1 = record_key(row)
    k2 = record_key(dict(row))
    assert k1 == k2
    other = dict(row); other['market'] = 'Hándicap'
    assert record_key(other) != k1


def test_unconfigured_google_is_fail_soft():
    old_id = os.environ.pop('GOOGLE_SHEETS_ID', None)
    old_json = os.environ.pop('GOOGLE_SERVICE_ACCOUNT_JSON', None)
    old_streamlit = sys.modules.pop('streamlit', None)
    try:
        status = sync_rows([{
            'game_date':'2026-08-25','game_pk':1,'market':'Moneyline',
            'selection':'Gana Local (NYY)','model_version':'v6'
        }], config={})
        assert status['ok'] is True
        assert status['configured'] is False
        assert status['inserted'] == 0
    finally:
        if old_id is not None: os.environ['GOOGLE_SHEETS_ID'] = old_id
        if old_json is not None: os.environ['GOOGLE_SERVICE_ACCOUNT_JSON'] = old_json
        if old_streamlit is not None: sys.modules['streamlit'] = old_streamlit


def test_streamlit_secrets_are_read_when_env_is_empty():
    old_id = os.environ.pop('GOOGLE_SHEETS_ID', None)
    old_json = os.environ.pop('GOOGLE_SERVICE_ACCOUNT_JSON', None)
    old_ws = os.environ.pop('GOOGLE_SHEETS_WORKSHEET', None)
    old_streamlit = sys.modules.get('streamlit')
    payload = {
        'type':'service_account', 'project_id':'mlb-test', 'private_key_id':'x',
        'private_key':'-----BEGIN PRIVATE KEY-----\nTEST\n-----END PRIVATE KEY-----\n',
        'client_email':'mlb-test@example.iam.gserviceaccount.com', 'client_id':'1',
        'token_uri':'https://oauth2.googleapis.com/token'
    }
    import json
    sys.modules['streamlit'] = SimpleNamespace(secrets={
        'GOOGLE_SHEETS_ID':'sheet-from-streamlit',
        'GOOGLE_SHEETS_WORKSHEET':'MLB_Picks',
        'GOOGLE_SERVICE_ACCOUNT_JSON':json.dumps(payload),
    })
    try:
        assert _sheet_id({}) == 'sheet-from-streamlit'
        assert _credentials_payload({})['project_id'] == 'mlb-test'
    finally:
        if old_id is not None: os.environ['GOOGLE_SHEETS_ID'] = old_id
        if old_json is not None: os.environ['GOOGLE_SERVICE_ACCOUNT_JSON'] = old_json
        if old_ws is not None: os.environ['GOOGLE_SHEETS_WORKSHEET'] = old_ws
        if old_streamlit is not None:
            sys.modules['streamlit'] = old_streamlit
        else:
            sys.modules.pop('streamlit', None)


def test_header_mismatch_empty_sheet_is_repairable():
    assert _schema_action([]) == 'reset'
    assert _schema_action([['Columna equivocada']]) == 'reset'


def test_header_mismatch_with_existing_data_is_preserved():
    assert _schema_action([['Viejo encabezado'], ['dato importante']]) == 'fallback'


def test_correct_header_is_accepted():
    assert _schema_action([SHEET_HEADERS]) == 'ok'


def test_sheet_schema_keeps_tracking_fields():
    required = {'record_key','game_date','game_pk','market','selection','odds','prob_ml','prob_mc','prob_combined','edge_pp','ev_pct','model_version','result_status','result_value','profit_units'}
    assert required.issubset(set(SHEET_HEADERS))


def main():
    tests=[v for k,v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for fn in tests:
        fn(); print('PASS', fn.__name__)
    print(f'Google Sheets tracking tests passed: {len(tests)}')


if __name__ == '__main__':
    main()
