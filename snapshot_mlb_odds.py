"""Collect timestamped pregame MLB odds for future edge/EV validation.

Primary source is the already-integrated TheRundown adapter. Nothing here changes
production picks. The archive is append-only and only stores games within eight
hours of first pitch, which is enough to reconstruct useful near-close prices
without letting the CSV grow unnecessarily.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import os
import pandas as pd
import requests

from modules.therundown_odds import _fetch_therundown
from modules.web_service import american_to_decimal

OUT=Path('data/mlb_odds_history.csv')
COLS=['snapshot_utc','event_id','commence_time_utc','home','away','book','market','selection','line','odds_american','odds_decimal']
MAX_HOURS_TO_START=8.0

def rows_from_event(ev, stamp_dt):
    commence=pd.to_datetime(ev.get('commence_time'),utc=True,errors='coerce')
    if pd.isna(commence): return []
    delta=(commence-pd.Timestamp(stamp_dt)).total_seconds()/3600.0
    if delta < 0 or delta > MAX_HOURS_TO_START: return []
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

def main():
    if not os.getenv('THERUNDOWN_KEY','').strip():
        print('::warning::THERUNDOWN_KEY no configurada; no se puede archivar cuotas reales.')
        return 0
    events=_fetch_therundown(requests.get)
    stamp_dt=datetime.now(timezone.utc)
    rows=[]
    for ev in events: rows.extend(rows_from_event(ev,stamp_dt))
    if not rows:
        print('Sin juegos MLB dentro de la ventana pregame de 8h o sin cuotas utilizables.')
        return 0
    new=pd.DataFrame(rows,columns=COLS)
    if OUT.exists():
        old=pd.read_csv(OUT)
        for c in COLS:
            if c not in old.columns: old[c]=None
        all_df=pd.concat([old[COLS],new],ignore_index=True)
    else: all_df=new
    keys=['snapshot_utc','event_id','book','market','selection','line']
    all_df=all_df.drop_duplicates(subset=keys,keep='last').sort_values(['snapshot_utc','commence_time_utc','event_id','book','market','selection'])
    OUT.parent.mkdir(parents=True,exist_ok=True); all_df.to_csv(OUT,index=False)
    print(f'ODDS_SNAPSHOT events_seen={len(events)} rows_added={len(new)} archive_rows={len(all_df)} at={stamp_dt.isoformat()}')
    return 0
if __name__=='__main__': raise SystemExit(main())
