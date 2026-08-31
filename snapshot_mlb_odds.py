"""Collect timestamped pregame MLB odds for future edge/EV validation.

Preferred source is the already-integrated TheRundown adapter when its secret is
available to the workflow. If not, the collector can read the public Cloud Run
/api/slate endpoint, which already exposes the normalized consensus market used
by production. Nothing here changes production picks.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import os
import pandas as pd
import requests

from modules.therundown_odds import _fetch_therundown
from modules.web_service import american_to_decimal

OUT = Path('data/mlb_odds_history.csv')
COLS = ['snapshot_utc','event_id','commence_time_utc','home','away','book','market','selection','line','odds_american','odds_decimal']
MAX_HOURS_TO_START = 8.0


def _american_from_decimal(value):
    try:
        dec=float(value)
        if dec <= 1.0: return None
        return round((dec-1.0)*100.0) if dec >= 2.0 else round(-100.0/(dec-1.0))
    except Exception:
        return None


def _in_window(commence_value, stamp_dt):
    commence=pd.to_datetime(commence_value,utc=True,errors='coerce')
    if pd.isna(commence): return False
    delta=(commence-pd.Timestamp(stamp_dt)).total_seconds()/3600.0
    return 0 <= delta <= MAX_HOURS_TO_START


def rows_from_event(ev, stamp_dt):
    if not _in_window(ev.get('commence_time'), stamp_dt): return []
    stamp=stamp_dt.isoformat().replace('+00:00','Z')
    out=[]; home=str(ev.get('home_team') or ''); away=str(ev.get('away_team') or '')
    for book in ev.get('bookmakers',[]) or []:
        title=str(book.get('title') or book.get('key') or 'unknown')
        for market in book.get('markets',[]) or []:
            key=str(market.get('key') or '')
            for o in market.get('outcomes',[]) or []:
                american=o.get('price'); dec=american_to_decimal(american)
                if dec is None: continue
                out.append({'snapshot_utc':stamp,'event_id':str(ev.get('id') or ''),'commence_time_utc':ev.get('commence_time'),'home':home,'away':away,'book':title,'market':key,'selection':str(o.get('name') or ''),'line':o.get('point'),'odds_american':american,'odds_decimal':dec})
    return out


def rows_from_production_game(game, stamp_dt):
    commence=game.get('start_time_utc')
    if not _in_window(commence, stamp_dt): return []
    stamp=stamp_dt.isoformat().replace('+00:00','Z')
    event_id=str(game.get('game_pk') or f"{game.get('away')}@{game.get('home')}:{commence}")
    home=str(game.get('home') or ''); away=str(game.get('away') or '')
    book='Production consensus'
    rows=[]
    def add(market, selection, odds, line=None):
        try: dec=float(odds)
        except Exception: return
        if dec <= 1.0: return
        rows.append({'snapshot_utc':stamp,'event_id':event_id,'commence_time_utc':commence,'home':home,'away':away,'book':book,'market':market,'selection':selection,'line':line,'odds_american':_american_from_decimal(dec),'odds_decimal':dec})
    add('h2h',home,game.get('cuota_loc')); add('h2h',away,game.get('cuota_vis'))
    line=game.get('linea_carreras'); add('totals','Over',game.get('cuota_over'),line); add('totals','Under',game.get('cuota_under'),line)
    add('spreads',home,game.get('cuota_spread_loc'),game.get('spread_loc')); add('spreads',away,game.get('cuota_spread_vis'),game.get('spread_vis'))
    return rows


def _production_rows(stamp_dt):
    base=os.getenv('MLB_PROD_URL','').strip().rstrip('/')
    if not base: return []
    r=requests.get(f'{base}/api/slate',timeout=30); r.raise_for_status()
    payload=r.json(); games=payload.get('games',[]) if isinstance(payload,dict) else []
    rows=[]
    for game in games or []:
        if isinstance(game,dict): rows.extend(rows_from_production_game(game,stamp_dt))
    print(f'PRODUCTION_SLATE games_seen={len(games)} rows_eligible={len(rows)}')
    return rows


def main():
    stamp_dt=datetime.now(timezone.utc)
    rows=[]; source='none'; events_seen=0
    if os.getenv('THERUNDOWN_KEY','').strip():
        events=_fetch_therundown(requests.get); events_seen=len(events); source='therundown'
        for ev in events: rows.extend(rows_from_event(ev,stamp_dt))
    if not rows and os.getenv('MLB_PROD_URL','').strip():
        rows=_production_rows(stamp_dt); source='production_consensus'
    if not rows:
        print('Sin cuotas MLB pregame utilizables; no se agrega snapshot.')
        return 0
    new=pd.DataFrame(rows,columns=COLS)
    if OUT.exists():
        old=pd.read_csv(OUT)
        for c in COLS:
            if c not in old.columns: old[c]=None
        all_df=pd.concat([old[COLS],new],ignore_index=True)
    else:
        all_df=new
    keys=['snapshot_utc','event_id','book','market','selection','line']
    all_df=all_df.drop_duplicates(subset=keys,keep='last').sort_values(['snapshot_utc','commence_time_utc','event_id','book','market','selection'])
    OUT.parent.mkdir(parents=True,exist_ok=True); all_df.to_csv(OUT,index=False)
    print(f'ODDS_SNAPSHOT source={source} events_seen={events_seen} rows_added={len(new)} archive_rows={len(all_df)} at={stamp_dt.isoformat()}')
    return 0

if __name__=='__main__': raise SystemExit(main())
