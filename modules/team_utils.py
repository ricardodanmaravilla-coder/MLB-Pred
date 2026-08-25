TEAM_MAP = {
    "New York Yankees": "NYY", "NYY": "NYY",
    "Boston Red Sox": "BOS", "BOS": "BOS",
    "Los Angeles Dodgers": "LAD", "LAD": "LAD",
    "Houston Astros": "HOU", "HOU": "HOU",
    "Atlanta Braves": "ATL", "ATL": "ATL",
    "Philadelphia Phillies": "PHI", "PHI": "PHI",
    "Baltimore Orioles": "BAL", "BAL": "BAL",
    "Tampa Bay Rays": "TB", "TB": "TB", "TBR": "TB",
    "Toronto Blue Jays": "TOR", "TOR": "TOR",
    "Chicago White Sox": "CWS", "CWS": "CWS", "CHW": "CWS",
    "Cleveland Guardians": "CLE", "CLE": "CLE",
    "Detroit Tigers": "DET", "DET": "DET",
    "Kansas City Royals": "KC", "KC": "KC", "KCR": "KC",
    "Minnesota Twins": "MIN", "MIN": "MIN",
    "Los Angeles Angels": "LAA", "LAA": "LAA", "ANA": "LAA",
    "Oakland Athletics": "OAK", "Athletics": "OAK", "Sacramento Athletics": "OAK",
    "OAK": "OAK", "ATH": "OAK",
    "Seattle Mariners": "SEA", "SEA": "SEA",
    "Texas Rangers": "TEX", "TEX": "TEX",
    "Chicago Cubs": "CHC", "CHC": "CHC",
    "Cincinnati Reds": "CIN", "CIN": "CIN",
    "Milwaukee Brewers": "MIL", "MIL": "MIL",
    "Pittsburgh Pirates": "PIT", "PIT": "PIT",
    "St. Louis Cardinals": "STL", "STL": "STL",
    "Arizona Diamondbacks": "AZ", "AZ": "AZ", "ARI": "AZ",
    "Colorado Rockies": "COL", "COL": "COL",
    "San Francisco Giants": "SF", "SF": "SF", "SFG": "SF",
    "San Diego Padres": "SD", "SD": "SD", "SDP": "SD",
    "Miami Marlins": "MIA", "MIA": "MIA",
    "New York Mets": "NYM", "NYM": "NYM",
    "Washington Nationals": "WSH", "WSH": "WSH", "WSN": "WSH",
}


def normalize_team(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text in TEAM_MAP:
        return TEAM_MAP[text]
    folded = text.casefold()
    for key, val in TEAM_MAP.items():
        if key.casefold() == folded:
            return val
    return text.upper() if len(text) <= 3 else text
