import os
import requests
import pandas as pd


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


def minar_stats_pitchers(season=None):
    season = int(season or pd.Timestamp.utcnow().year)
    print(f"⚾ [INICIO] Extrayendo pitchers MLB {season} desde StatsAPI...")
    os.makedirs('data', exist_ok=True)
    starters_path = 'data/mlb_pitching_individual.csv'
    bullpen_path = 'data/mlb_bullpen.csv'

    url = f"https://statsapi.mlb.com/api/v1/stats?stats=season&season={season}&group=pitching&playerPool=ALL&limit=2000&sportId=1"
    headers = {'User-Agent': 'MLB-Pred/3.0'}
    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        blocks = response.json().get('stats', [])
        splits = blocks[0].get('splits', []) if blocks else []
        rows = []
        for split in splits:
            player = split.get('player', {})
            team = split.get('team', {})
            stat = split.get('stat', {})
            name = player.get('fullName')
            team_abbr = team.get('abbreviation') or team.get('name')
            era_raw = stat.get('era')
            if not name or not team_abbr or era_raw in (None, '-.--'):
                continue
            try:
                era = float(era_raw)
            except (TypeError, ValueError):
                continue
            rows.append({
                'Name': name,
                'Team': team_abbr,
                'ERA': era,
                # Legacy compatibility only. This is NOT real xFIP.
                'xFIP': era,
                'xFIP_Source': 'LEGACY_ERA_NOT_REAL_XFIP',
                'GS': int(stat.get('gamesStarted', 0) or 0),
                'G': int(stat.get('gamesPlayed', 0) or 0),
                'IP': _innings_to_float(stat.get('inningsPitched', 0)),
                'Season': season,
            })

        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError('StatsAPI no devolvió lanzadores utilizables')

        starters = df[df['GS'] > 0].copy()
        starters.to_csv(starters_path, index=False)
        print(f"✅ Abridores/swingmen guardados: {len(starters)}")

        # A true role-split bullpen requires relief-only splits. StatsAPI season
        # totals do not provide that here, so use only pitchers with zero starts.
        # This is substantially cleaner than reusing full team ERA, and it is
        # explicitly labeled as a reliever-only proxy.
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
        bullpen = pd.DataFrame(bullpen_rows)
        bullpen.to_csv(bullpen_path, index=False)
        print(f"✅ Bullpen proxy separado guardado: {len(bullpen)} equipos")
        return starters, bullpen
    except Exception as e:
        print(f"❌ Error crítico en descarga de pitchers: {e}")
        return pd.DataFrame(), pd.DataFrame()


if __name__ == '__main__':
    minar_stats_pitchers()
