import os
import requests
import pandas as pd


BASE = 'https://statsapi.mlb.com/api/v1'
HEADERS = {'User-Agent': 'MLB-Pred/3.1'}


def _innings_to_float(value):
    """Convert baseball IP notation (12.1 = 12 1/3, 12.2 = 12 2/3)."""
    try:
        text = str(value or '0')
        whole, _, frac = text.partition('.')
        outs = int(frac[:1] or 0)
        if outs not in (0, 1, 2):
            return float(text)
        return float(int(whole) + outs / 3.0)
    except Exception:
        return 0.0


def _get_json(url, timeout=20):
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _mlb_teams():
    data = _get_json(f'{BASE}/teams?sportId=1')
    out = []
    for t in data.get('teams', []):
        tid = t.get('id')
        abbr = t.get('abbreviation') or t.get('teamCode') or t.get('fileCode') or t.get('name')
        if tid and abbr:
            out.append((int(tid), str(abbr).upper()))
    return out


def minar_stats_pitchers(season=None):
    """Persist starters plus a reliever-only bullpen proxy from official MLB StatsAPI.

    The global player stats endpoint does not reliably include team metadata, so V3
    queries each MLB team explicitly. This prevents UNK/missing-team pitcher rows.
    `xFIP` remains only a legacy compatibility alias for ERA and is labeled as such.
    """
    season = int(season or pd.Timestamp.utcnow().year)
    print(f'⚾ [INICIO] Extrayendo pitchers MLB {season} por equipo...')
    os.makedirs('data', exist_ok=True)
    starters_path = 'data/mlb_pitching_individual.csv'
    bullpen_path = 'data/mlb_bullpen.csv'

    rows = []
    failures = []
    try:
        teams = _mlb_teams()
        if len(teams) < 25:
            raise RuntimeError(f'StatsAPI devolvió solo {len(teams)} equipos MLB')

        for team_id, team_abbr in teams:
            url = (f'{BASE}/stats?stats=season&season={season}&group=pitching&'
                   f'playerPool=ALL&limit=250&sportId=1&teamId={team_id}')
            try:
                blocks = _get_json(url).get('stats', [])
                splits = blocks[0].get('splits', []) if blocks else []
                for split in splits:
                    player = split.get('player', {})
                    stat = split.get('stat', {})
                    name = player.get('fullName')
                    era_raw = stat.get('era')
                    if not name or era_raw in (None, '-.--'):
                        continue
                    try:
                        era = float(era_raw)
                    except (TypeError, ValueError):
                        continue
                    rows.append({
                        'Name': str(name),
                        'Team': team_abbr,
                        'ERA': era,
                        'xFIP': era,
                        'xFIP_Source': 'LEGACY_ERA_NOT_REAL_XFIP',
                        'GS': int(stat.get('gamesStarted', 0) or 0),
                        'G': int(stat.get('gamesPlayed', 0) or 0),
                        'IP': _innings_to_float(stat.get('inningsPitched', 0)),
                        'Season': season,
                    })
            except Exception as e:
                failures.append((team_abbr, str(e)))

        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError('StatsAPI no devolvió lanzadores utilizables')
        df = df.drop_duplicates(subset=['Name', 'Team'], keep='last')

        starters = df[df['GS'] > 0].copy().sort_values(['Team','ERA','Name'])
        starters.to_csv(starters_path, index=False)

        # Reliever-only proxy: pitchers with zero starts, weighted by innings.
        # It deliberately excludes swingmen so starter innings are not counted again.
        relievers = df[(df['GS'] == 0) & (df['IP'] >= 3.0)].copy()
        bullpen_rows = []
        for team_code, grp in relievers.groupby('Team'):
            ip = float(grp['IP'].sum())
            if ip <= 0:
                continue
            era_weighted = float((grp['ERA'] * grp['IP']).sum() / ip)
            bullpen_rows.append({
                'Team': team_code,
                'ERA': round(era_weighted, 3),
                'IP': round(ip, 1),
                'Relievers': int(len(grp)),
                'Season': season,
                'Source': 'RELIEVER_ZERO_STARTS_IP_WEIGHTED_PROXY',
            })
        bullpen = pd.DataFrame(bullpen_rows).sort_values('Team') if bullpen_rows else pd.DataFrame()
        if len(bullpen) < 20:
            raise RuntimeError(f'Bullpen proxy incompleto: solo {len(bullpen)} equipos')
        bullpen.to_csv(bullpen_path, index=False)

        print(f'✅ Abridores/swingmen guardados: {len(starters)}')
        print(f'✅ Bullpen proxy separado guardado: {len(bullpen)} equipos')
        if failures:
            print(f'⚠️ Equipos con error parcial: {failures}')
        return starters, bullpen
    except Exception as e:
        print(f'❌ Error crítico en descarga de pitchers: {e}')
        return pd.DataFrame(), pd.DataFrame()


if __name__ == '__main__':
    minar_stats_pitchers()
