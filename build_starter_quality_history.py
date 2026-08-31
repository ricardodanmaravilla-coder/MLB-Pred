"""Build validation-only MLB starter game logs from official completed-game feeds."""
from __future__ import annotations
import concurrent.futures as cf
import time
from pathlib import Path
import pandas as pd
import requests

GAMES=Path('data/mlb_games.csv'); OUT=Path('data/mlb_starter_quality_history.csv')
URL='https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live'

def _num(x):
    try:return float(x)
    except:return 0.0

def _ip(x):
    s=str(x or '0'); a,b=(s.split('.')+['0'])[:2]
    try:return float(a)+float(b)/3.0
    except:return 0.0

def _fetch(pk,date):
    for attempt in range(3):
        try:
            r=requests.get(URL.format(game_pk=pk),timeout=20); r.raise_for_status(); j=r.json()
            if ((j.get('gameData') or {}).get('status') or {}).get('abstractGameState') not in ('Final',None): return [],None
            teams=(((j.get('liveData') or {}).get('boxscore') or {}).get('teams') or {}); players=((j.get('gameData') or {}).get('players') or {}); rows=[]
            for side in ('away','home'):
                t=teams.get(side,{}) or {}; ids=t.get('pitchers') or []
                if not ids: continue
                pid=int(ids[0]); p=(t.get('players') or {}).get(f'ID{pid}',{}) or {}; st=((p.get('stats') or {}).get('pitching') or {}); gp=players.get(f'ID{pid}',{}) or {}
                rows.append({'GameID':int(pk),'Date':date,'Side':side,'Team':(t.get('team') or {}).get('abbreviation'),'StarterID':pid,'StarterName':gp.get('fullName') or (p.get('person') or {}).get('fullName'),'PitchHand':(gp.get('pitchHand') or {}).get('code'),'IP':_ip(st.get('inningsPitched')),'H':_num(st.get('hits')),'ER':_num(st.get('earnedRuns')),'BB':_num(st.get('baseOnBalls')),'SO':_num(st.get('strikeOuts')),'HR':_num(st.get('homeRuns')),'BF':_num(st.get('battersFaced')),'Pitches':_num(st.get('numberOfPitches')),'Source':'MLB_OFFICIAL_LIVE_FEED_BOXSCORE'})
            return rows,None
        except Exception as e:
            if attempt==2:return [],str(e)
            time.sleep(.5*(attempt+1))

def main():
    g=pd.read_csv(GAMES,low_memory=False); idc=next((c for c in ('GameID','gamePk','game_id') if c in g.columns),None); g['Date']=pd.to_datetime(g.Date,errors='coerce'); g=g[g.Date.dt.year>=2025].dropna(subset=['Date',idc]); jobs=[(int(r[idc]),r.Date.date().isoformat()) for _,r in g.iterrows()]; rows=[]; fail=[]
    print(f'Descargando calidad oficial de abridores para {len(jobs)} juegos...')
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        fs={ex.submit(_fetch,pk,d):pk for pk,d in jobs}
        for i,f in enumerate(cf.as_completed(fs),1):
            rr,e=f.result(); rows.extend(rr); fail.append((fs[f],e)) if e else None
            if i%500==0 or i==len(jobs): print(f'  procesados={i}/{len(jobs)} rows={len(rows)} fallos={len(fail)}')
    x=pd.DataFrame(rows); OUT.parent.mkdir(parents=True,exist_ok=True); x.to_csv(OUT,index=False); cov=(x.groupby('GameID').size().ge(2).sum()/max(len(jobs),1)) if not x.empty else 0
    print(f'OK {OUT} rows={len(x)} coverage={cov:.1%} failures={len(fail)}')
    if cov<.90: raise SystemExit(f'Insufficient starter quality coverage: {cov:.1%}')
if __name__=='__main__':main()
