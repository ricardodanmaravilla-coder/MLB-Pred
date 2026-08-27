import os
import requests
import pandas as pd

from modules.advanced_stats import enrich_pitcher_frame, fetch_fangraphs_team_season
from modules.team_utils import normalize_team

BASE='https://statsapi.mlb.com/api/v1';HEADERS={'User-Agent':'MLB-Pred/7.0'}


def _innings_to_float(value):
    try:
        text=str(value or '0');whole,_,frac=text.partition('.');outs=int(frac[:1] or 0)
        if outs not in (0,1,2):return float(text)
        return float(int(whole)+outs/3.0)
    except Exception:return 0.0

def _get_json(url,timeout=20):
    r=requests.get(url,headers=HEADERS,timeout=timeout);r.raise_for_status();return r.json()

def _mlb_teams():
    data=_get_json(f'{BASE}/teams?sportId=1');out=[]
    for t in data.get('teams',[]):
        tid=t.get('id');raw=t.get('abbreviation') or t.get('teamCode') or t.get('fileCode') or t.get('name');abbr=normalize_team(raw)
        if tid and abbr:out.append((int(tid),str(abbr).upper()))
    return out

def _pitch_hands(player_ids):
    out={};ids=[int(x) for x in sorted(set(player_ids)) if x]
    for i in range(0,len(ids),75):
        batch=ids[i:i+75]
        try:
            data=_get_json(f"{BASE}/people?personIds={','.join(map(str,batch))}")
            for p in data.get('people',[]):
                pid=p.get('id');hand=(p.get('pitchHand') or {}).get('code') or (p.get('pitchHand') or {}).get('description')
                if pid and hand:out[int(pid)]=str(hand).upper()[0]
        except Exception as exc:print(f'⚠️ PitchHand batch no disponible: {exc}')
    return out

def _fangraphs_relievers(season):
    try:_,_,rel=fetch_fangraphs_team_season(int(season))
    except Exception as exc:print(f'⚠️ Bullpen FanGraphs no disponible: {exc}');return pd.DataFrame()
    if rel is None or rel.empty:return pd.DataFrame()
    rows=[]
    for _,r in rel.iterrows():
        team=normalize_team(r.get('Team'))
        try:era=float(r.get('ERA'))
        except (TypeError,ValueError):continue
        row={'Team':team,'ERA':era,'Season':int(season),'Source':'FANGRAPHS_REAL_RELIEVERS'}
        for c in ('xFIP','FIP','SIERA','K-BB%','K%','BB%','WHIP','GB%','HR/9','IP'):
            try:row[c]=float(r.get(c))
            except (TypeError,ValueError):row[c]=None
        rows.append(row)
    return pd.DataFrame(rows).dropna(subset=['Team']).drop_duplicates('Team',keep='last') if rows else pd.DataFrame()

def minar_stats_pitchers(season=None):
    season=int(season or pd.Timestamp.utcnow().year);print(f'⚾ [INICIO] Extrayendo pitchers MLB {season} por equipo...');os.makedirs('data',exist_ok=True);starters_path='data/mlb_pitching_individual.csv';bullpen_path='data/mlb_bullpen.csv';rows,failures=[],[]
    try:
        teams=_mlb_teams()
        if len(teams)<25:raise RuntimeError(f'StatsAPI devolvió solo {len(teams)} equipos MLB')
        for team_id,team_abbr in teams:
            url=(f'{BASE}/stats?stats=season&season={season}&group=pitching&playerPool=ALL&limit=250&sportId=1&teamId={team_id}')
            try:
                blocks=_get_json(url).get('stats',[]);splits=blocks[0].get('splits',[]) if blocks else []
                for split in splits:
                    player=split.get('player',{});stat=split.get('stat',{});name=player.get('fullName');pid=player.get('id');era_raw=stat.get('era')
                    if not name or era_raw in (None,'-.--'):continue
                    try:era=float(era_raw)
                    except (TypeError,ValueError):continue
                    g=int(stat.get('gamesPlayed',0) or 0);gs=int(stat.get('gamesStarted',0) or 0);ip=_innings_to_float(stat.get('inningsPitched',0));relief_apps=max(0,g-gs);relief_share=(relief_apps/g) if g>0 else 0.0
                    rows.append({'PlayerID':pid,'Name':str(name),'Team':team_abbr,'ERA':era,'xFIP':era,'xFIP_Source':'LEGACY_ERA_NOT_REAL_XFIP','FIP':None,'SIERA':None,'K-BB%':None,'K%':None,'BB%':None,'WHIP':stat.get('whip'),'GS':gs,'G':g,'IP':ip,'ReliefApps':relief_apps,'ReliefShare':round(relief_share,4),'Season':season})
            except Exception as e:failures.append((team_abbr,str(e)))
        df=pd.DataFrame(rows)
        if df.empty:raise RuntimeError('StatsAPI no devolvió lanzadores utilizables')
        hands=_pitch_hands(pd.to_numeric(df['PlayerID'],errors='coerce').dropna().astype(int).tolist());df['PitchHand']=pd.to_numeric(df['PlayerID'],errors='coerce').map(lambda x:hands.get(int(x)) if pd.notna(x) else None);df=df.drop_duplicates(subset=['PlayerID','Team'],keep='last');df=enrich_pitcher_frame(df,season)
        starters=df[df['GS']>0].copy();sort_metric='xFIP' if 'xFIP' in starters.columns else 'ERA';starters=starters.sort_values(['Team',sort_metric,'Name']);starters.to_csv(starters_path,index=False)
        bullpen_fg=_fangraphs_relievers(season)
        if bullpen_fg is not None and len(bullpen_fg)>=25:bullpen=bullpen_fg.sort_values('Team')
        else:
            relief=df[(df['ReliefApps']>=3)&(df['IP']>=3.0)].copy();relief['ReliefIPProxy']=relief['IP']*relief['ReliefShare'];relief=relief[relief['ReliefIPProxy']>=1.0];bullpen_rows=[]
            for team_code,grp in relief.groupby('Team'):
                ip=float(grp['ReliefIPProxy'].sum())
                if ip<=0:continue
                era_weighted=float((pd.to_numeric(grp['ERA'],errors='coerce')*grp['ReliefIPProxy']).sum()/ip);bullpen_rows.append({'Team':normalize_team(team_code),'ERA':round(era_weighted,3),'xFIP':None,'FIP':None,'SIERA':None,'IP':round(ip,1),'Relievers':int(len(grp)),'Season':season,'Source':'RELIEF_APPEARANCE_SHARE_WEIGHTED_ERA_PROXY'})
            bullpen=pd.DataFrame(bullpen_rows).drop_duplicates('Team',keep='last').sort_values('Team') if bullpen_rows else pd.DataFrame()
        if len(bullpen)<25:raise RuntimeError(f'Bullpen incompleto: solo {len(bullpen)} equipos')
        bullpen.to_csv(bullpen_path,index=False);real_starters=int(starters.get('xFIP_Source',pd.Series(dtype=str)).astype(str).str.contains('FANGRAPHS_REAL').sum());hand_cov=round(float(starters['PitchHand'].notna().mean())*100,1) if 'PitchHand' in starters else 0.0
        print(f'✅ Abridores/swingmen guardados: {len(starters)}; xFIP real: {real_starters}; mano conocida: {hand_cov}%');print(f"✅ Bullpen guardado: {len(bullpen)} equipos; fuente={bullpen['Source'].iloc[0] if 'Source' in bullpen else 'unknown'}")
        if failures:print(f'⚠️ Equipos con error parcial: {failures}')
        return starters,bullpen
    except Exception as e:print(f'❌ Error crítico en descarga de pitchers: {e}');return pd.DataFrame(),pd.DataFrame()

if __name__=='__main__':minar_stats_pitchers()
