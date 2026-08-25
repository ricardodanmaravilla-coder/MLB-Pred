import os
import numpy as np
import pandas as pd
import requests

from modules.team_utils import normalize_team

BASE='https://statsapi.mlb.com/api/v1'; HEADERS={'User-Agent':'MLB-Pred/6.0'}

def _innings_to_float(value):
    try:
        text=str(value or '0'); whole,_,frac=text.partition('.'); outs=int(frac[:1] or 0)
        if outs not in (0,1,2):return float(text)
        return float(int(whole)+outs/3.)
    except Exception:return 0.
def _f(v,default=0.):
    try:return float(v) if v not in (None,'','-.--') else float(default)
    except (TypeError,ValueError):return float(default)
def _get_json(url,timeout=20):
    r=requests.get(url,headers=HEADERS,timeout=timeout);r.raise_for_status();return r.json()
def _mlb_teams():
    out=[]
    for t in _get_json(f'{BASE}/teams?sportId=1').get('teams',[]):
        tid=t.get('id');abbr=normalize_team(t.get('abbreviation') or t.get('teamCode') or t.get('name'))
        if tid and abbr:out.append((int(tid),str(abbr).upper()))
    return out

def _statsapi_pitchers(season):
    rows=[];failures=[];teams=_mlb_teams()
    if len(teams)<25:raise RuntimeError(f'StatsAPI devolvió solo {len(teams)} equipos MLB')
    for team_id,team in teams:
        url=f'{BASE}/stats?stats=season&season={season}&group=pitching&playerPool=ALL&limit=300&sportId=1&teamId={team_id}'
        try:
            blocks=_get_json(url).get('stats',[]);splits=blocks[0].get('splits',[]) if blocks else []
            for split in splits:
                player,stat=split.get('player',{}),split.get('stat',{});name=player.get('fullName');era_raw=stat.get('era')
                if not name or era_raw in (None,'-.--'):continue
                era=_f(era_raw,np.nan)
                if pd.isna(era):continue
                g=int(stat.get('gamesPlayed',0) or 0);gs=int(stat.get('gamesStarted',0) or 0);ip=_innings_to_float(stat.get('inningsPitched',0));relief_apps=max(0,g-gs);relief_share=relief_apps/g if g>0 else 0.
                rows.append({'Name':str(name),'Team':team,'ERA':era,'GS':gs,'G':g,'IP':ip,'ReliefApps':relief_apps,'ReliefShare':round(relief_share,4),'Season':season,
                             'HR':_f(stat.get('homeRuns')),'BB':_f(stat.get('baseOnBalls')),'HBP':_f(stat.get('hitBatsmen',stat.get('hitByPitch'))),'K':_f(stat.get('strikeOuts')),'BF':_f(stat.get('battersFaced')),
                             'WHIP':_f(stat.get('whip'),np.nan),'DataSource':'MLB_STATSAPI_OFFICIAL'})
        except Exception as e:failures.append((team,str(e)))
    df=pd.DataFrame(rows)
    if df.empty:return df,failures
    ip=float(df['IP'].sum());hr=float(df['HR'].sum());bb=float(df['BB'].sum());hbp=float(df['HBP'].sum());k=float(df['K'].sum())
    weights=df['IP'].clip(lower=.1);lg_era=float(np.average(df['ERA'],weights=weights));raw=(13*hr+3*(bb+hbp)-2*k)/ip if ip>0 else 0.;const=lg_era-raw
    rip=df['IP'].replace(0,np.nan);bf=df['BF'].replace(0,np.nan)
    df['FIP']=(13*df['HR']+3*(df['BB']+df['HBP'])-2*df['K'])/rip+const;df['FIP_Constant']=const;df['K-BB%']=(df['K']-df['BB'])/bf*100.;df['HR/9']=df['HR']/rip*9.;df['xFIP']=df['FIP'];df['xFIP_Source']='MLB_OFFICIAL_FIP_USED_NOT_XFIP'
    return df,failures

def minar_stats_pitchers(season=None):
    season=int(season or pd.Timestamp.utcnow().year);print(f'⚾ [INICIO] Pitchers MLB oficiales + FIP {season}...');os.makedirs('data',exist_ok=True)
    try:
        df,failures=_statsapi_pitchers(season)
        if df.empty:raise RuntimeError('StatsAPI no devolvió lanzadores utilizables')
        df=df.drop_duplicates(['Name','Team'],keep='last');starters=df[df['GS']>0].copy().sort_values(['Team','FIP','Name']);starters.to_csv('data/mlb_pitching_individual.csv',index=False)
        relief=df[(df['ReliefApps']>=3)&(df['IP']>=3.)].copy();relief['ReliefIPProxy']=relief['IP']*relief['ReliefShare'];relief=relief[relief['ReliefIPProxy']>=1.]
        rows=[]
        for team,grp in relief.groupby('Team'):
            weights=grp['ReliefIPProxy'].clip(lower=.1);valid=pd.to_numeric(grp['FIP'],errors='coerce').notna()
            if not valid.any():continue
            w=weights[valid]; fip=float(np.average(grp.loc[valid,'FIP'],weights=w)); era=float(np.average(grp.loc[valid,'ERA'],weights=w)); kbb=float(np.average(grp.loc[valid,'K-BB%'].fillna(0),weights=w)); hr9=float(np.average(grp.loc[valid,'HR/9'].fillna(0),weights=w)); whip=float(np.average(grp.loc[valid,'WHIP'].fillna(grp.loc[valid,'WHIP'].median()),weights=w)) if grp.loc[valid,'WHIP'].notna().any() else np.nan
            rows.append({'Team':normalize_team(team),'ERA':round(fip,3),'FIP':round(fip,3),'ERA_Actual':round(era,3),'K-BB%':round(kbb,3),'HR/9':round(hr9,3),'WHIP':None if pd.isna(whip) else round(whip,3),'IP':round(float(w.sum()),1),'Relievers':int(len(grp)),'Season':season,'Source':'MLB_OFFICIAL_RELIEF_SHARE_WEIGHTED_FIP'})
        bullpen=pd.DataFrame(rows).drop_duplicates('Team',keep='last').sort_values('Team') if rows else pd.DataFrame()
        if len(bullpen)<25:raise RuntimeError(f'Bullpen incompleto: solo {len(bullpen)} equipos')
        bullpen.to_csv('data/mlb_bullpen.csv',index=False)
        print(f'✅ Pitchers oficiales/FIP: {len(starters)}; bullpen FIP: {len(bullpen)} equipos')
        if failures:print(f'⚠️ Errores parciales StatsAPI: {failures}')
        return starters,bullpen
    except Exception as e:
        print(f'❌ Error crítico pitchers: {e}');return pd.DataFrame(),pd.DataFrame()

if __name__=='__main__':minar_stats_pitchers()
