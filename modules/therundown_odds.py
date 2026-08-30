from __future__ import annotations

"""TheRundown V2 MLB odds provider for the existing multi-odds bridge.

The provider is intentionally isolated from the scanner so the current
The-Odds-API-shaped pipeline does not need to change.  It fetches only the
full-game main-line moneyline, spread and total markets for MLB and normalizes
them to the schema already consumed by modules.multi_odds.

Environment / Streamlit secret:
  THERUNDOWN_KEY            required API key
  THERUNDOWN_AFFILIATE_IDS  optional CSV, default DraftKings, BetMGM, FanDuel
"""

import os
from collections import defaultdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from . import multi_odds

_INSTALLED = False
_MLB_SPORT_ID = 3
_CENTRAL = ZoneInfo("America/Chicago")
_AFFILIATE_NAMES = {
    "2": "Bovada",
    "3": "Pinnacle",
    "4": "Sportsbetting",
    "6": "BetOnline",
    "11": "LowVig",
    "12": "Bodog",
    "14": "Intertops",
    "16": "Matchbook",
    "18": "YouWager",
    "19": "DraftKings",
    "21": "Unibet",
    "22": "BetMGM",
    "23": "FanDuel",
    "24": "theScore Bet",
    "25": "Kalshi",
    "26": "Polymarket",
}


def _secret(name: str, default: str = "") -> str:
    return multi_odds._secret(name, default)


def _team_name(team: dict) -> str:
    name = str(team.get("name") or "").strip()
    mascot = str(team.get("mascot") or "").strip()
    if mascot and mascot.lower() not in name.lower():
        return f"{name} {mascot}".strip()
    return name or mascot


def _price(value):
    try:
        v = float(value)
        # TheRundown uses 0.0001 as an off-board sentinel.
        if abs(v - 0.0001) < 1e-9:
            return None
        if abs(v) < 100:
            return None
        return round(v)
    except (TypeError, ValueError):
        return None


def _float(value):
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fetch_therundown(get_fn) -> list[dict]:
    key = _secret("THERUNDOWN_KEY")
    if not key:
        return []

    slate_date = datetime.now(timezone.utc).astimezone(_CENTRAL).date().isoformat()
    affiliate_ids = _secret("THERUNDOWN_AFFILIATE_IDS", "19,22,23")
    url = f"https://therundown.io/api/v2/sports/{_MLB_SPORT_ID}/events/{slate_date}"
    params = {
        "market_ids": "1,2,3",
        "affiliate_ids": affiliate_ids,
        "main_line": "true",
        "hide_closed": "true",
        "offset": "300",
    }
    headers = {"X-TheRundown-Key": key, "Accept": "application/json"}

    try:
        response = get_fn(url, params=params, headers=headers, timeout=10)
        if getattr(response, "status_code", 0) != 200:
            return []
        payload = response.json()
        events = payload.get("events", []) if isinstance(payload, dict) else []
        if not isinstance(events, list):
            return []

        normalized = []
        for event in events:
            if not isinstance(event, dict):
                continue

            teams = event.get("teams") or []
            if len(teams) < 2:
                continue
            away_obj = next((t for t in teams if t.get("is_away")), teams[0])
            home_obj = next((t for t in teams if t.get("is_home")), teams[1])
            away = _team_name(away_obj)
            home = _team_name(home_obj)
            if not home or not away:
                continue

            home_id = str(home_obj.get("team_id") or "")
            away_id = str(away_obj.get("team_id") or "")
            by_book = defaultdict(lambda: {
                "ml": {},
                "totals": defaultdict(dict),
                "spreads": {},
            })

            for market in event.get("markets") or []:
                if not isinstance(market, dict):
                    continue
                market_id = int(market.get("market_id") or 0)
                period_id = int(market.get("period_id") or 0)
                if market_id not in (1, 2, 3) or period_id != 0:
                    continue

                for participant in market.get("participants") or []:
                    if not isinstance(participant, dict):
                        continue
                    pid = str(participant.get("id") or "")
                    pname = str(participant.get("name") or "").strip()
                    if pid == home_id:
                        selection = home
                    elif pid == away_id:
                        selection = away
                    else:
                        selection = pname

                    for line in participant.get("lines") or []:
                        if not isinstance(line, dict):
                            continue
                        point = _float(line.get("value"))
                        prices = line.get("prices") or {}
                        if not isinstance(prices, dict):
                            continue

                        for affiliate_id, price_obj in prices.items():
                            if not isinstance(price_obj, dict):
                                continue
                            price = _price(price_obj.get("price"))
                            if price is None:
                                continue
                            aid = str(affiliate_id)
                            slot = by_book[aid]

                            if market_id == 1 and selection in (home, away):
                                slot["ml"][selection] = price
                            elif market_id == 2 and selection in (home, away) and point is not None:
                                slot["spreads"][selection] = {"point": point, "price": price}
                            elif market_id == 3 and point is not None:
                                side = "Over" if pname.lower().startswith("over") else "Under" if pname.lower().startswith("under") else None
                                if side:
                                    slot["totals"][point][side] = price

            books = []
            for aid, data in by_book.items():
                markets = []
                ml = data["ml"]
                if home in ml and away in ml:
                    markets.append(multi_odds._market("h2h", [
                        multi_odds._outcome(home, ml[home]),
                        multi_odds._outcome(away, ml[away]),
                    ]))

                valid_totals = [
                    (point, sides) for point, sides in data["totals"].items()
                    if "Over" in sides and "Under" in sides
                ]
                if valid_totals:
                    point, sides = valid_totals[0]
                    markets.append(multi_odds._market("totals", [
                        multi_odds._outcome("Over", sides["Over"], point),
                        multi_odds._outcome("Under", sides["Under"], point),
                    ]))

                spreads = data["spreads"]
                if home in spreads and away in spreads:
                    hp, ap = spreads[home], spreads[away]
                    if abs(float(hp["point"]) + float(ap["point"])) < 1e-6:
                        markets.append(multi_odds._market("spreads", [
                            multi_odds._outcome(home, hp["price"], hp["point"]),
                            multi_odds._outcome(away, ap["price"], ap["point"]),
                        ]))

                if markets:
                    title = _AFFILIATE_NAMES.get(aid, f"TheRundown {aid}")
                    books.append({
                        "key": f"therundown_{aid}",
                        "title": title,
                        "markets": markets,
                    })

            if books:
                normalized.append(multi_odds._event(
                    home,
                    away,
                    event.get("event_date"),
                    event.get("event_id") or event.get("event_uuid"),
                    books,
                ))

        return normalized
    except Exception:
        return []


def install_therundown_provider() -> None:
    """Attach TheRundown to multi_odds without changing the scanner contract."""
    global _INSTALLED
    if _INSTALLED:
        return

    original_sharp = multi_odds._fetch_sharpapi

    def fetch_alternates(get_fn):
        # Return separate normalized events. multi_odds performs the final merge
        # and creates one consensus snapshot across all available providers.
        return list(_fetch_therundown(get_fn)) + list(original_sharp(get_fn))

    multi_odds._fetch_sharpapi = fetch_alternates

    # The legacy application only enters its odds code path when ODDS_API_KEY is
    # non-empty.  If TheRundown is the only configured provider, use the same
    # harmless sentinel understood by the existing requests bridge.
    if _secret("THERUNDOWN_KEY") and not os.getenv("ODDS_API_KEY", "").strip():
        os.environ["ODDS_API_KEY"] = multi_odds._SENTINEL

    _INSTALLED = True
