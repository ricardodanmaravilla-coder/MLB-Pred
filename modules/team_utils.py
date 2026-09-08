TEAM_MAP = {
    "New York Yankees": "NYY", "Yankees": "NYY", "NYY": "NYY",
    "Boston Red Sox": "BOS", "Red Sox": "BOS", "BOS": "BOS",
    "Los Angeles Dodgers": "LAD", "LA Dodgers": "LAD", "Dodgers": "LAD", "LAD": "LAD",
    "Houston Astros": "HOU", "Astros": "HOU", "HOU": "HOU",
    "Atlanta Braves": "ATL", "Braves": "ATL", "ATL": "ATL",
    "Philadelphia Phillies": "PHI", "Phillies": "PHI", "PHI": "PHI",
    "Baltimore Orioles": "BAL", "Orioles": "BAL", "BAL": "BAL",
    "Tampa Bay Rays": "TB", "Rays": "TB", "TB": "TB", "TBR": "TB",
    "Toronto Blue Jays": "TOR", "Blue Jays": "TOR", "TOR": "TOR",
    "Chicago White Sox": "CWS", "White Sox": "CWS", "CWS": "CWS", "CHW": "CWS",
    "Cleveland Guardians": "CLE", "Guardians": "CLE", "CLE": "CLE",
    "Detroit Tigers": "DET", "Tigers": "DET", "DET": "DET",
    "Kansas City Royals": "KC", "Royals": "KC", "KC": "KC", "KCR": "KC",
    "Minnesota Twins": "MIN", "Twins": "MIN", "MIN": "MIN",
    "Los Angeles Angels": "LAA", "LA Angels": "LAA", "Angels": "LAA", "LAA": "LAA", "ANA": "LAA",
    "Oakland Athletics": "OAK", "Sacramento Athletics": "OAK", "Athletics": "OAK", "A's": "OAK", "As": "OAK",
    "OAK": "OAK", "ATH": "OAK",
    "Seattle Mariners": "SEA", "Mariners": "SEA", "SEA": "SEA",
    "Texas Rangers": "TEX", "Rangers": "TEX", "TEX": "TEX",
    "Chicago Cubs": "CHC", "Cubs": "CHC", "CHC": "CHC",
    "Cincinnati Reds": "CIN", "Reds": "CIN", "CIN": "CIN",
    "Milwaukee Brewers": "MIL", "Brewers": "MIL", "MIL": "MIL",
    "Pittsburgh Pirates": "PIT", "Pirates": "PIT", "PIT": "PIT",
    "St. Louis Cardinals": "STL", "St Louis Cardinals": "STL", "Cardinals": "STL", "STL": "STL",
    "Arizona Diamondbacks": "AZ", "Diamondbacks": "AZ", "D-backs": "AZ", "Dbacks": "AZ", "AZ": "AZ", "ARI": "AZ",
    "Colorado Rockies": "COL", "Rockies": "COL", "COL": "COL",
    "San Francisco Giants": "SF", "Giants": "SF", "SF": "SF", "SFG": "SF",
    "San Diego Padres": "SD", "Padres": "SD", "SD": "SD", "SDP": "SD",
    "Miami Marlins": "MIA", "Marlins": "MIA", "MIA": "MIA",
    "New York Mets": "NYM", "NY Mets": "NYM", "Mets": "NYM", "NYM": "NYM",
    "Washington Nationals": "WSH", "Nationals": "WSH", "Nats": "WSH", "WSH": "WSH", "WSN": "WSH",
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
