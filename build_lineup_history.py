"""Build official MLB lineups, hitter logs, starter hands and platoon game logs.

Validation-only datasets. All hitter-vs-hand rows come from official completed-game
play-by-play. Starter hand comes from the official game feed. No missing split, hand,
or batting result is fabricated; downstream validation uses only rows strictly before
the prediction date.
"""
from __future__ import annotations
import concurrent.futures as cf
import time
from collections import defaultdict
from pathlib import Path
import pandas as pd
import requests

GAMES=Path('data/mlb_games.csv')
LINEUPS_OUT=Path('data/mlb_lineup_history.csv')
HITTERS_OUT=Path('data/mlb_hitter_game_history.csv')
PLATOON_OUT=Path('data/mlb_hitter_platoon_game_history.csv')
STARTERS_OUT=Path('data/mlb_starter_hand_history.csv')
URL='https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live'
BAT_KEYS={'atBats':'AB','hits':'H','doubles':'2B','triples':'3B','homeRuns':'HR','baseOnBalls':'BB','strikeOuts':'SO','hitByPitch':'HBP','sacFlies':'SF'}
STAT_COLS=('AB','H','2B','3B','HR','BB','SO','HBP','SF')
AB_EVENTS={'single','double','triple','home_run','field_out','force_out','grounded_into_double_play','field_error','strikeout','strikeout_double_play','fielders_choice','fielders_choice_out','double_play','triple_play'}


def _event_stats(event):
    z={c:0 for c in STAT_COLS}; e=str(event or '').lower()
    if e in AB_EVENTS: z['AB']=1
    if e=='single': z['H']=1
    elif e=='double': z['H']=1; z['2B']=1
    elif e=='triple': z['H']=1; z['3B']=1
    elif e=='home_run': z['H']=1; z['HR']=1
    elif e in {'walk','intent_walk'}: z['BB']=1
    elif e=='hit_by_pitch': z['HBP']=1
    elif e=='sac_fly': z['SF']=1
    if e.startswith('strikeout'): z['SO']=1
    return z


def _hand(players,pid,play=None):
    h=((play or {}).get('matchup') or {}).get('pitchHand') or {}
    code=h.get('code')
    if code in ('R','L'): return code
    p=players.get(f'ID{pid}',{}) or {}; code=(p.get('pitchHand') or {}).get('code')
    return code if code in ('R','L') else None


def _fetch(game_pk:int,date:str):
    for attempt in range(3):
        try:
            r=requests.get(URL.format(game_pk=game_pk),timeout=20)
            if r.status_code!=200: raise RuntimeError(f'HTTP {r.status_code}')
            j=r.json(); status=((j.get('gameData') or {}).get('status') or {}).get('abstractGameState')
            if status not in ('Final',None): return [],[],[],[],None
            lineup_rows=[]; hitter_rows=[]; starter_rows=[]; platoon=defaultdict(lambda:{c:0 for c in STAT_COLS})
            box=(j.get('liveData') or {}).get('boxscore') or {}; teams=box.get('teams') or {}; gd=j.get('gameData') or {}; players=gd.get('players') or {}
            team_names={}
            for side in ('away','home'):
                t=teams.get(side,{}) or {}; team=(t.get('team') or {}).get('abbreviation') or (t.get('team') or {}).get('name'); team_names[side]=team
                bplayers=t.get('players') or {}; order=t.get('battingOrder') or []
                if len(order)>=9:
                    for slot,pid in enumerate(order[:9],1):
                        p=bplayers.get(f'ID{pid}',{}) or {}; person=p.get('person') or {}; pos=p.get('position') or {}
                        lineup_rows.append({'GameID':int(game_pk),'Date':date,'Side':side,'Team':team,'BattingOrder':slot,'PlayerID':int(pid),'PlayerName':person.get('fullName'),'Position':pos.get('abbreviation'),'Source':'MLB_OFFICIAL_LIVE_FEED'})
                pitchers=t.get('pitchers') or []
                if pitchers:
                    pid=int(pitchers[0]); gp=players.get(f'ID{pid}',{}) or {}; hand=(gp.get('pitchHand') or {}).get('code')
                    if hand not in ('R','L'):
                        bp=bplayers.get(f'ID{pid}',{}) or {}; hand=(bp.get('pitchHand') or {}).get('code')
                    starter_rows.append({'GameID':int(game_pk),'Date':date,'Side':side,'Team':team,'StarterID':pid,'StarterName':(gp.get('fullName') or (bplayers.get(f'ID{pid}',{}).get('person') or {}).get('fullName')),'PitchHand':hand if hand in ('R','L') else None,'Source':'MLB_OFFICIAL_LIVE_FEED'})
                for p in bplayers.values():
                    person=p.get('person') or {}; pid=person.get('id'); batting=((p.get('stats') or {}).get('batting') or {})
                    if pid is None or not batting: continue
                    vals={out:batting.get(src) for src,out in BAT_KEYS.items()}; nums=[pd.to_numeric(vals.get(c),errors='coerce') for c in ('AB','BB','HBP','SF')]
                    if sum(float(x) if pd.notna(x) else 0. for x in nums)<=0: continue
                    hitter_rows.append({'GameID':int(game_pk),'Date':date,'Side':side,'Team':team,'PlayerID':int(pid),'PlayerName':person.get('fullName'),**vals,'Source':'MLB_OFFICIAL_LIVE_FEED_BOXSCORE'})
            for play in ((j.get('liveData') or {}).get('plays') or {}).get('allPlays',[]):
                matchup=play.get('matchup') or {}; batter=(matchup.get('batter') or {}).get('id'); pitcher=(matchup.get('pitcher') or {}).get('id')
                if batter is None or pitcher is None: continue
                hand=_hand(players,int(pitcher),play)
                if hand not in ('R','L'): continue
                event=(play.get('result') or {}).get('eventType'); vals=_event_stats(event)
                if sum(vals.values())<=0: continue
                half=((play.get('about') or {}).get('halfInning') or '').lower(); side='away' if half=='top' else 'home' if half=='bottom' else None
                key=(int(batter),hand,side)
                for c,v in vals.items(): platoon[key][c]+=v
            platoon_rows=[]
            for (pid,hand,side),vals in platoon.items():
                pa=vals['AB']+vals['BB']+vals['HBP']+vals['SF']
                if pa<=0: continue
                gp=players.get(f'ID{pid}',{}) or {}
                platoon_rows.append({'GameID':int(game_pk),'Date':date,'Side':side,'Team':team_names.get(side),'PlayerID':pid,'PlayerName':gp.get('fullName'),'PitcherHand':hand,**vals,'Source':'MLB_OFFICIAL_LIVE_FEED_PLAYBYPLAY'})
            return lineup_rows,hitter_rows,platoon_rows,starter_rows,None
        except Exception as e:
            if attempt==2: return [],[],[],[],str(e)
            time.sleep(.5*(attempt+1))


def main():
    g=pd.read_csv(GAMES,low_memory=False); idcol=next((c for c in ('GameID','gamePk','game_id') if c in g.columns),None)
    if not idcol or 'Date' not in g.columns: raise SystemExit('mlb_games.csv requires GameID/gamePk and Date')
    g['Date']=pd.to_datetime(g['Date'],errors='coerce'); g=g[g.Date.dt.year>=2025].dropna(subset=['Date',idcol]).copy()
    jobs=[(int(r[idcol]),r.Date.date().isoformat()) for _,r in g.iterrows()]
    lineups=[]; hitters=[]; platoon=[]; starters=[]; failures=[]
    print(f'Descargando datos oficiales de lineup/platoon para {len(jobs)} juegos MLB desde 2025...')
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        futs={ex.submit(_fetch,pk,d):pk for pk,d in jobs}
        for i,fut in enumerate(cf.as_completed(futs),1):
            lr,hr,pr,sr,err=fut.result(); lineups.extend(lr); hitters.extend(hr); platoon.extend(pr); starters.extend(sr)
            if err: failures.append((futs[fut],err))
            if i%500==0 or i==len(jobs): print(f'  procesados={i}/{len(jobs)} lineups={len(lineups)} platoon={len(platoon)} starters={len(starters)} fallos={len(failures)}')
    dfs=[pd.DataFrame(lineups),pd.DataFrame(hitters),pd.DataFrame(platoon),pd.DataFrame(starters)]
    LINEUPS_OUT.parent.mkdir(parents=True,exist_ok=True)
    for df,path in zip(dfs,(LINEUPS_OUT,HITTERS_OUT,PLATOON_OUT,STARTERS_OUT)): df.to_csv(path,index=False)
    lu,hi,pl,st=dfs; complete=0 if lu.empty else int((lu.groupby(['GameID','Side']).size()>=9).groupby(level=0).sum().ge(2).sum()); coverage=complete/max(len(jobs),1)
    starter_complete=0 if st.empty else int((st.dropna(subset=['PitchHand']).groupby('GameID').size()>=2).sum()); starter_cov=starter_complete/max(len(jobs),1)
    platoon_games=0 if pl.empty else int(pl.GameID.nunique())
    print(f'OK {LINEUPS_OUT} rows={len(lu)} complete_games={complete} coverage={coverage:.1%}')
    print(f'OK {HITTERS_OUT} rows={len(hi)} games={0 if hi.empty else hi.GameID.nunique()}')
    print(f'OK {PLATOON_OUT} rows={len(pl)} games={platoon_games}')
    print(f'OK {STARTERS_OUT} rows={len(st)} starter_hand_coverage={starter_cov:.1%} failures={len(failures)}')
    if coverage<.85: raise SystemExit(f'Insufficient official lineup coverage: {coverage:.1%}')
    if starter_cov<.85: raise SystemExit(f'Insufficient official starter-hand coverage: {starter_cov:.1%}')
    if platoon_games/max(len(jobs),1)<.85: raise SystemExit('Insufficient official platoon play-by-play coverage')

if __name__=='__main__': main()
