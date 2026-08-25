import os
import requests
import pandas as pd

from modules.fangraphs_mlb import fetch_pitcher_fangraphs, fetch_team_fangraphs
from modules.team_utils import normalize_team

BASE = 'https://statsapi.mlb.com/api/v1'
HEADERS = {'User-Agent': 'MLB-Pred/6.0'}


def _innings_to_float(value):
    try:
        text = str(value or '0'); whole, _, frac = text.partition('.'); outs = int(frac[:1] or 0)
        if outs not in (0, 1, 2): return float(text)
        return float(int(whole) + outs / 3.0)
    except Exception: return 0.0


def _get_json(url, timeout=20):
    r = requests.get(url, headers=HEADERS, timeout=timeout); r.raise_for_status(); return r.json()


def _mlb_teams():
    data = _get_json(f'{BASE}/teams?sportId=1'); out = []
    for t in data.get('teams', []):
        tid = t.get('id'); raw = t.get('abbreviation') or t.get('teamCode') or t.get('fileCode') or t.get('name'); abbr = normalize_team(raw)
        if tid and abbr: out.append((int(tid), str(abbr).upper()))
    return out


def _statsapi_pitchers(season):
    rows, failures = [], []
    teams = _mlb_teams()
    if len(teams) < 25: raise RuntimeError(f'StatsAPI devolvió solo {len(teams)} equipos MLB')
    for team_id, team_abbr in teams:
        url = (f'{BASE}/stats?stats=season&season={season}&group=pitching&playerPool=ALL&limit=250&sportId=1&teamId={team_id}')
        try:
            blocks = _get_json(url).get('stats', []); splits = blocks[0].get('splits', []) if blocks else []
            for split in splits:
                player, stat = split.get('player', {}), split.get('stat', {}); name = player.get('fullName'); era_raw = stat.get('era')
                if not name or era_raw in (None, '-.--'): continue
                try: era = float(era_raw)
                except (TypeError, ValueError): continue
                g = int(stat.get('gamesPlayed', 0) or 0); gs = int(stat.get('gamesStarted', 0) or 0); ip = _innings_to_float(stat.get('inningsPitched', 0))
                relief_apps = max(0, g - gs); relief_share = (relief_apps / g) if g > 0 else 0.0
                rows.append({'Name': str(name), 'Team': team_abbr, 'ERA': era, 'xFIP': era, 'xFIP_Source': 'LEGACY_ERA_NOT_REAL_XFIP',
                             'GS': gs, 'G': g, 'IP': ip, 'ReliefApps': relief_apps, 'ReliefShare': round(relief_share, 4), 'Season': season,
                             'DataSource': 'MLB_STATSAPI_FALLBACK'})
        except Exception as e: failures.append((team_abbr, str(e)))
    return pd.DataFrame(rows), failures


def _merge_fangraphs_players(base, fg):
    if base.empty or fg.empty: return base
    b = base.copy(); f = fg.copy()
    b['_name'] = b['Name'].astype(str).str.casefold().str.replace(r'[^a-z0-9 ]','',regex=True).str.strip()
    f['_name'] = f['Name'].astype(str).str.casefold().str.replace(r'[^a-z0-9 ]','',regex=True).str.strip()
    cols = ['_name','Team'] + [c for c in ('xFIP','FIP','WHIP','K/9','BB/9','HR/9','K%','BB%','K-BB%','GB%') if c in f.columns]
    f = f[cols].drop_duplicates(['_name','Team'], keep='last')
    out = b.merge(f, on=['_name','Team'], how='left', suffixes=('','_FG'))
    for c in ('xFIP','FIP','WHIP','K/9','BB/9','HR/9','K%','BB%','K-BB%','GB%'):
        fgcol = c + '_FG'
        if fgcol in out.columns:
            if c in out.columns: out[c] = pd.to_numeric(out[fgcol], errors='coerce').combine_first(pd.to_numeric(out[c], errors='coerce'))
            else: out[c] = pd.to_numeric(out[fgcol], errors='coerce')
            out.drop(columns=[fgcol], inplace=True)
    real = out['xFIP'].notna() & out['_name'].isin(set(f['_name']))
    out.loc[real, 'xFIP_Source'] = 'FANGRAPHS_REAL_XFIP'; out.loc[real, 'DataSource'] = 'MLB_STATSAPI_PLUS_FANGRAPHS'
    return out.drop(columns=['_name'])


def minar_stats_pitchers(season=None):
    season = int(season or pd.Timestamp.utcnow().year); print(f'⚾ [INICIO] Pitchers MLB/FanGraphs {season}...'); os.makedirs('data', exist_ok=True)
    try:
        raw, failures = _statsapi_pitchers(season)
        if raw.empty: raise RuntimeError('StatsAPI no devolvió lanzadores utilizables')
        fg_players = fetch_pitcher_fangraphs(season)
        df = _merge_fangraphs_players(raw.drop_duplicates(['Name','Team'], keep='last'), fg_players)
        starters = df[df['GS'] > 0].copy().sort_values(['Team','ERA','Name']); starters.to_csv('data/mlb_pitching_individual.csv', index=False)

        # Prefer true FanGraphs reliever aggregate because it already separates role.
        _, _, fg_rel = fetch_team_fangraphs(season, season)
        if not fg_rel.empty and 'xFIP' in fg_rel.columns:
            bullpen = fg_rel.copy(); bullpen['Team'] = bullpen['Team'].map(normalize_team); bullpen['Season'] = season
            bullpen['ERA_Estimator'] = pd.to_numeric(bullpen['xFIP'], errors='coerce').combine_first(pd.to_numeric(bullpen.get('FIP'), errors='coerce')).combine_first(pd.to_numeric(bullpen.get('ERA'), errors='coerce'))
            bullpen['ERA'] = bullpen['ERA_Estimator']; bullpen['Source'] = 'FANGRAPHS_REAL_RELIEVER_XFIP'
            bullpen = bullpen.dropna(subset=['Team','ERA']).drop_duplicates('Team', keep='last')
            keep = [c for c in ('Team','ERA','xFIP','FIP','WHIP','K-BB%','HR/9','GB%','IP','Season','Source') if c in bullpen.columns]
            bullpen = bullpen[keep].sort_values('Team')
        else:
            relief = df[(df['ReliefApps'] >= 3) & (df['IP'] >= 3.0)].copy(); relief['ReliefIPProxy'] = relief['IP'] * relief['ReliefShare']; relief = relief[relief['ReliefIPProxy'] >= 1.0]
            rows = []
            for team_code, grp in relief.groupby('Team'):
                ip = float(grp['ReliefIPProxy'].sum())
                if ip <= 0: continue
                metric = pd.to_numeric(grp['xFIP'], errors='coerce').combine_first(pd.to_numeric(grp['ERA'], errors='coerce'))
                valid = metric.notna(); weights = grp.loc[valid,'ReliefIPProxy']
                if not valid.any() or weights.sum() <= 0: continue
                est = float((metric[valid] * weights).sum() / weights.sum())
                rows.append({'Team': normalize_team(team_code), 'ERA': round(est,3), 'IP': round(ip,1), 'Relievers': int(len(grp)), 'Season': season,
                             'Source': 'RELIEF_APPEARANCE_SHARE_WEIGHTED_XFIP_OR_ERA'})
            bullpen = pd.DataFrame(rows).drop_duplicates('Team', keep='last').sort_values('Team') if rows else pd.DataFrame()
        if len(bullpen) < 25: raise RuntimeError(f'Bullpen incompleto: solo {len(bullpen)} equipos')
        bullpen.to_csv('data/mlb_bullpen.csv', index=False)
        print(f'✅ Pitchers: {len(starters)}; bullpen avanzado: {len(bullpen)} equipos')
        if failures: print(f'⚠️ Errores parciales StatsAPI: {failures}')
        return starters, bullpen
    except Exception as e:
        print(f'❌ Error crítico pitchers: {e}'); return pd.DataFrame(), pd.DataFrame()


if __name__ == '__main__': minar_stats_pitchers()
