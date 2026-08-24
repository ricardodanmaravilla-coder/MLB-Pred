TEAM_NAME_TO_ABBR = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL",
    "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Oakland Athletics": "OAK", "Athletics": "OAK",
    "Sacramento Athletics": "OAK", "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDP", "San Francisco Giants": "SFG", "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WSN",
}

# Aliases comunes entre MLB StatsAPI, casas y datasets históricos.
ABBR_ALIASES = {
    "AZ": "ARI", "ARI": "ARI", "CWS": "CHW", "CHW": "CHW", "KC": "KC", "KCR": "KC",
    "SD": "SDP", "SDP": "SDP", "SF": "SFG", "SFG": "SFG", "TB": "TB", "TBR": "TB",
    "WSH": "WSN", "WSN": "WSN", "OAK": "OAK", "ATH": "OAK",
}


def normalize_team(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text in TEAM_NAME_TO_ABBR:
        return TEAM_NAME_TO_ABBR[text]
    up = text.upper()
    return ABBR_ALIASES.get(up, up if 2 <= len(up) <= 3 else None)
