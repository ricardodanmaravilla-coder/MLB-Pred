from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from .team_utils import normalize_team

SLATE_TZ = ZoneInfo("America/New_York")
ROOF_OR_DOME_TEAMS = {"AZ", "HOU", "MIA", "MIL", "SEA", "TB", "TEX", "TOR"}
# Ballpark/city coordinates, sufficient for hourly outdoor weather selection.
BALLPARK_COORDS = {
    'NYY':(40.8296,-73.9262),'BOS':(42.3467,-71.0972),'LAD':(34.0739,-118.2400),'HOU':(29.7573,-95.3555),
    'ATL':(33.8908,-84.4677),'PHI':(39.9061,-75.1665),'BAL':(39.2838,-76.6217),'TB':(27.7682,-82.6534),
    'TOR':(43.6414,-79.3894),'CWS':(41.8300,-87.6339),'CLE':(41.4962,-81.6852),'DET':(42.3390,-83.0485),
    'KC':(39.0517,-94.4803),'MIN':(44.9817,-93.2776),'LAA':(33.8003,-117.8827),'OAK':(38.5802,-121.4997),
    'SEA':(47.5914,-122.3325),'TEX':(32.7473,-97.0847),'CHC':(41.9484,-87.6553),'CIN':(39.0979,-84.5082),
    'MIL':(43.0280,-87.9712),'PIT':(40.4469,-80.0057),'STL':(38.6226,-90.1928),'AZ':(33.4455,-112.0667),
    'COL':(39.7559,-104.9942),'SF':(37.7786,-122.3893),'SD':(32.7073,-117.1573),'MIA':(25.7781,-80.2197),
    'NYM':(40.7571,-73.8458),'WSH':(38.8730,-77.0074),
}


def slate_date(now=None):
    if now is None: now=datetime.now(timezone.utc)
    if now.tzinfo is None: now=now.replace(tzinfo=timezone.utc)
    return now.astimezone(SLATE_TZ).date()


def parse_utc(value):
    if value in (None,""): return None
    try:
        dt=pd.to_datetime(value,utc=True,errors='coerce')
        if pd.isna(dt): return None
        return dt.to_pydatetime()
    except Exception: return None


def hourly_weather_for_game(team,start_time_utc,timeout=8):
    """Fetch nearest-hour outdoor forecast from Open-Meteo (no API key required)."""
    target=normalize_team(team)
    if target in ROOF_OR_DOME_TEAMS: return 72,0,'None','neutral_roof_unknown'
    coords=BALLPARK_COORDS.get(target); start=parse_utc(start_time_utc)
    if coords is None or start is None: return None,None,'None','forecast_unavailable'
    try:
        lat,lon=coords
        params={'latitude':lat,'longitude':lon,'hourly':'temperature_2m,wind_speed_10m,wind_direction_10m','temperature_unit':'fahrenheit','wind_speed_unit':'mph','timezone':'UTC','forecast_days':7}
        r=requests.get('https://api.open-meteo.com/v1/forecast',params=params,timeout=timeout); r.raise_for_status(); hourly=r.json().get('hourly',{})
        times=pd.to_datetime(hourly.get('time',[]),utc=True,errors='coerce')
        if len(times)==0: return None,None,'None','forecast_unavailable'
        diffs=abs(times-pd.Timestamp(start)); idx=int(diffs.argmin())
        if pd.isna(times[idx]) or abs((times[idx].to_pydatetime()-start).total_seconds())>5400: return None,None,'None','forecast_unavailable'
        temp=float(hourly['temperature_2m'][idx]); wind=float(hourly['wind_speed_10m'][idx]); deg=hourly.get('wind_direction_10m',[None]*len(times))[idx]
        direction='None' if deg is None else f'Compass {float(deg):.0f}°'
        return temp,wind,direction,'open_meteo_hourly'
    except Exception as exc:
        print(f'Hourly weather unavailable for {target}: {exc}'); return None,None,'None','forecast_unavailable'


def conservative_auto_weather(team,start_time_utc,temp_f,wind_mph,wind_dir,now=None):
    target=normalize_team(team)
    if target in ROOF_OR_DOME_TEAMS: return 72,0,"None","neutral_roof_unknown"
    start=parse_utc(start_time_utc); current=now or datetime.now(timezone.utc)
    if current.tzinfo is None: current=current.replace(tzinfo=timezone.utc)
    if start is None or abs((start-current).total_seconds())>2*3600: return 72,0,"None","neutral_not_near_first_pitch"
    if temp_f is None or wind_mph is None: return 72,0,"None","neutral_weather_unavailable"
    return temp_f,wind_mph,wind_dir or "None","current_near_first_pitch"


def best_auto_weather(team,start_time_utc,current_temp=None,current_wind=None,current_dir=None):
    """Prefer game-time hourly forecast, then conservative near-first-pitch current data."""
    temp,wind,direction,source=hourly_weather_for_game(team,start_time_utc)
    if source in ('open_meteo_hourly','neutral_roof_unknown'): return temp,wind,direction,source
    return conservative_auto_weather(team,start_time_utc,current_temp,current_wind,current_dir)


def park_for_team(df_parks,team):
    if df_parks is None or df_parks.empty:return None
    cols=list(df_parks.columns); team_col=next((c for c in ['Team','TeamCode','Abbr','Franchise','Equipo','franchise'] if c in cols),None); pf_col=next((c for c in cols if 'park_factor' in c.lower() or c.lower() in ('factor','pf')),None); alt_col=next((c for c in cols if 'altitud' in c.lower() or 'elevation' in c.lower() or c.lower() in ('alt','altitude')),None)
    if team_col is None or pf_col is None or alt_col is None:return None
    target=normalize_team(team); keys=df_parks[team_col].map(normalize_team); rows=df_parks[keys==target]
    if rows.empty:return None
    row=rows.iloc[-1]
    try:return {'team':target,'park_factor':float(row[pf_col]),'altitude_ft':float(row[alt_col]),'stadium':row.get('Estadio',row.get('Stadium','Unknown'))}
    except (TypeError,ValueError):return None

def _same_matchup(odds_game,mlb_game):return normalize_team(odds_game.get('home_team'))==normalize_team(mlb_game.get('local')) and normalize_team(odds_game.get('away_team'))==normalize_team(mlb_game.get('visita'))
def match_odds_game(odds_games,mlb_game,max_hours=2.0):
    candidates=[g for g in (odds_games or []) if _same_matchup(g,mlb_game)]
    if not candidates:return None
    target=parse_utc(mlb_game.get('start_time_utc'))
    if target is None:return candidates[0] if len(candidates)==1 else None
    scored=[]
    for g in candidates:
        dt=parse_utc(g.get('commence_time'))
        if dt is not None:scored.append((abs((dt-target).total_seconds())/3600.,g))
    if not scored:return None
    scored.sort(key=lambda x:x[0])
    if scored[0][0]>float(max_hours):return None
    if len(scored)>1 and abs(scored[1][0]-scored[0][0])<.25:return None
    return scored[0][1]
def market_from_event(event,american_to_decimal):
    empty={'linea_carreras':None,'cuota_loc':None,'cuota_vis':None,'cuota_over':None,'cuota_under':None,'spread_loc':None,'cuota_spread_loc':None,'spread_vis':None,'cuota_spread_vis':None,'bookmaker':None}
    if not event:return dict(empty)
    home,away=event.get('home_team'),event.get('away_team'); snapshots=[]
    for bookmaker in event.get('bookmakers',[]):
        snap=dict(empty); found=set()
        for market in bookmaker.get('markets',[]):
            key=market.get('key'); outcomes=market.get('outcomes',[])
            if key=='h2h':
                for o in outcomes:
                    name=normalize_team(o.get('name'))
                    if name==normalize_team(home):snap['cuota_loc']=american_to_decimal(o.get('price'))
                    elif name==normalize_team(away):snap['cuota_vis']=american_to_decimal(o.get('price'))
                if snap['cuota_loc'] and snap['cuota_vis']:found.add('h2h')
            elif key=='totals':
                op=up=None
                for o in outcomes:
                    if o.get('name')=='Over':op=o.get('point');snap['cuota_over']=american_to_decimal(o.get('price'))
                    elif o.get('name')=='Under':up=o.get('point');snap['cuota_under']=american_to_decimal(o.get('price'))
                if op is not None and up is not None and float(op)==float(up) and snap['cuota_over'] and snap['cuota_under']:snap['linea_carreras']=float(op);found.add('totals')
            elif key=='spreads':
                for o in outcomes:
                    name=normalize_team(o.get('name'));point=o.get('point')
                    if name==normalize_team(home) and point is not None:snap['spread_loc']=float(point);snap['cuota_spread_loc']=american_to_decimal(o.get('price'))
                    elif name==normalize_team(away) and point is not None:snap['spread_vis']=float(point);snap['cuota_spread_vis']=american_to_decimal(o.get('price'))
                if snap['spread_loc'] is not None and snap['spread_vis'] is not None and abs(snap['spread_loc']+snap['spread_vis'])<1e-9 and snap['cuota_spread_loc'] and snap['cuota_spread_vis']:found.add('spreads')
                else:snap['spread_loc']=snap['cuota_spread_loc']=None;snap['spread_vis']=snap['cuota_spread_vis']=None
        if 'h2h' in found:snap['bookmaker']=bookmaker.get('title') or bookmaker.get('key');snapshots.append((len(found),snap))
    if not snapshots:return dict(empty)
    snapshots.sort(key=lambda x:x[0],reverse=True);return snapshots[0][1]
