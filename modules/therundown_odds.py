from __future__ import annotations

"""TheRundown V2 MLB odds provider for the existing multi-odds bridge.

The provider is intentionally isolated from the scanner so the current
The-Odds-API-shaped pipeline does not need to change. It fetches the full-game
moneyline, spread and total markets for MLB and normalizes them to the schema
already consumed by modules.multi_odds.

Environment / Streamlit secret:
  THERUNDOWN_KEY            required API key
  THERUNDOWN_AFFILIATE_IDS  optional CSV, default DraftKings, BetMGM, FanDuel
"""

import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from . import multi_odds

log = logging.getLogger(__name__)

_INSTALLED = False
# The /api/v2/sports catalog for this account reports MLB as sport_id=3.
_MLB_SPORT_ID = 3
_CENTRAL = ZoneInfo("America/Chicago")
_CACHE_TTL_SECONDS = 300
_CACHE: dict[str, object] = {"date": None, "at": 0.0, "events": []}

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


def _norm_name(value) -> str:
    return " ".join(str(value or "").lower().replace(".", "").split())


def _same_team(a: str, b: str) -> bool:
    """Tolerate provider differences such as 'Orioles' vs 'Baltimore Orioles'."""
    na, nb = _norm_name(a), _norm_name(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    # Require a meaningful token match; this safely handles nickname-only names.
    ta, tb = na.split(), nb.split()
    return len(ta[-1]) >= 4 and ta[-1] == tb[-1]


def _event_teams(event: dict):
    teams = [t for t in (event.get("teams") or []) if isinstance(t, dict)]
    if len(teams) < 2:
        return None

    away_obj = next((t for t in teams if t.get("is_away") is True), None)
    home_obj = next((t for t in teams if t.get("is_home") is True), None)

    # V2 commonly exposes only is_away. Away is index 0 and home index 1.
    if away_obj is None:
        away_obj = teams[0]
    if home_obj is None:
        home_obj = next((t for t in teams if t is not away_obj and t.get("is_away") is False), None)
    if home_obj is None:
        home_obj = teams[1] if teams[1] is not away_obj else teams[0]

    away = _team_name(away_obj)
    home = _team_name(home_obj)
    if not away or not home:
        return None
    return home_obj, away_obj, home, away


def _team_selection(participant: dict, home_obj: dict, away_obj: dict, home: str, away: str):
    pid = str(participant.get("id") or participant.get("team_id") or "")
    # TheRundown V2 team objects use `id`; older payloads may use team_id.
    home_id = str(home_obj.get("id") or home_obj.get("team_id") or "")
    away_id = str(away_obj.get("id") or away_obj.get("team_id") or "")
    pname = str(participant.get("name") or "").strip()

    if pid and home_id and pid == home_id:
        return home
    if pid and away_id and pid == away_id:
        return away
    if _same_team(pname, home):
        return home
    if _same_team(pname, away):
        return away
    return pname


def _selected_affiliates() -> set[str]:
    raw = _secret("THERUNDOWN_AFFILIATE_IDS", "19,22,23")
    return {x.strip() for x in raw.split(",") if x.strip()}


def _request_payload(get_fn, url: str, headers: dict, affiliate_ids: str):
    """Use the narrow production request first; retry once with only core markets.

    The fallback exists because some account/sport catalog combinations have
    returned an empty event list when affiliate/main-line filters were combined,
    while the same MLB endpoint returned events without those filters.
    """
    narrow = {
        "market_ids": "1,2,3",
        "affiliate_ids": affiliate_ids,
        "main_line": "true",
        "hide_closed": "true",
        "offset": "300",
    }
    response = get_fn(url, params=narrow, headers=headers, timeout=12)
    if getattr(response, "status_code", 0) != 200:
        log.warning("TheRundown narrow request failed: HTTP %s", getattr(response, "status_code", None))
        return None

    payload = response.json()
    events = payload.get("events", []) if isinstance(payload, dict) else []
    if isinstance(events, list) and events:
        return payload

    # One bounded fallback: still request only the three core markets. We filter
    # affiliates and main lines locally. This is intentionally not an unfiltered
    # all-markets request, which could consume many data points.
    fallback = {"market_ids": "1,2,3", "offset": "300"}
    response = get_fn(url, params=fallback, headers=headers, timeout=12)
    if getattr(response, "status_code", 0) != 200:
        log.warning("TheRundown fallback request failed: HTTP %s", getattr(response, "status_code", None))
        return payload
    return response.json()


def _fetch_therundown(get_fn) -> list[dict]:
    key = _secret("THERUNDOWN_KEY")
    if not key:
        return []

    slate_date = datetime.now(timezone.utc).astimezone(_CENTRAL).date().isoformat()
    now = time.monotonic()
    if _CACHE.get("date") == slate_date and now - float(_CACHE.get("at") or 0.0) < _CACHE_TTL_SECONDS:
        return list(_CACHE.get("events") or [])

    affiliate_ids = _secret("THERUNDOWN_AFFILIATE_IDS", "19,22,23")
    allowed_affiliates = _selected_affiliates()
    url = f"https://therundown.io/api/v2/sports/{_MLB_SPORT_ID}/events/{slate_date}"
    headers = {"X-TheRundown-Key": key, "Accept": "application/json"}

    try:
        payload = _request_payload(get_fn, url, headers, affiliate_ids)
        events = payload.get("events", []) if isinstance(payload, dict) else []
        if not isinstance(events, list):
            return []

        normalized = []
        for event in events:
            if not isinstance(event, dict):
                continue

            team_info = _event_teams(event)
            if not team_info:
                continue
            home_obj, away_obj, home, away = team_info

            # Store all valid main-line candidates by book. A dict keyed by point
            # prevents alternate lines from overwriting the actual main line.
            by_book = defaultdict(lambda: {
                "ml": {},
                "totals": defaultdict(dict),
                "spreads": defaultdict(dict),
            })

            for market in event.get("markets") or []:
                if not isinstance(market, dict):
                    continue
                try:
                    market_id = int(market.get("market_id") or 0)
                    period_id = int(market.get("period_id") or 0)
                except (TypeError, ValueError):
                    continue
                if market_id not in (1, 2, 3) or period_id != 0:
                    continue

                for participant in market.get("participants") or []:
                    if not isinstance(participant, dict):
                        continue
                    pname = str(participant.get("name") or "").strip()
                    ptype = str(participant.get("type") or "").upper()
                    selection = _team_selection(participant, home_obj, away_obj, home, away)

                    for line in participant.get("lines") or []:
                        if not isinstance(line, dict):
                            continue
                        point = _float(line.get("value"))
                        prices = line.get("prices") or {}
                        if not isinstance(prices, dict):
                            continue

                        for affiliate_id, price_obj in prices.items():
                            aid = str(affiliate_id)
                            if allowed_affiliates and aid not in allowed_affiliates:
                                continue
                            if not isinstance(price_obj, dict):
                                continue

                            # In V2 is_main_line belongs to the price object. If the
                            # field is present and false, it is an alternate line.
                            if "is_main_line" in price_obj and price_obj.get("is_main_line") is not True:
                                continue
                            price = _price(price_obj.get("price"))
                            if price is None:
                                continue

                            slot = by_book[aid]
                            if market_id == 1 and selection in (home, away):
                                slot["ml"][selection] = price
                            elif market_id == 2 and selection in (home, away) and point is not None:
                                slot["spreads"][point][selection] = price
                            elif market_id == 3 and point is not None:
                                side = None
                                if ptype in {"TYPE_OVER", "OVER"} or pname.lower().startswith("over"):
                                    side = "Over"
                                elif ptype in {"TYPE_UNDER", "UNDER"} or pname.lower().startswith("under"):
                                    side = "Under"
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
                    # With is_main_line filtering there should normally be one.
                    point, sides = valid_totals[0]
                    markets.append(multi_odds._market("totals", [
                        multi_odds._outcome("Over", sides["Over"], point),
                        multi_odds._outcome("Under", sides["Under"], point),
                    ]))

                spread_market = None
                for point, selections in data["spreads"].items():
                    if home not in selections:
                        continue
                    opposite = -float(point)
                    away_rows = data["spreads"].get(opposite, {})
                    if away in away_rows:
                        spread_market = multi_odds._market("spreads", [
                            multi_odds._outcome(home, selections[home], point),
                            multi_odds._outcome(away, away_rows[away], opposite),
                        ])
                        break
                if spread_market:
                    markets.append(spread_market)

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
                    event.get("event_id") or event.get("event_uuid") or event.get("id"),
                    books,
                ))

        _CACHE.update({"date": slate_date, "at": now, "events": normalized})
        log.info("TheRundown normalized %d MLB events for %s", len(normalized), slate_date)
        return normalized
    except Exception:
        log.exception("TheRundown MLB odds normalization failed")
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
    # non-empty. If TheRundown is the only configured provider, use the same
    # harmless sentinel understood by the existing requests bridge.
    if _secret("THERUNDOWN_KEY") and not os.getenv("ODDS_API_KEY", "").strip():
        os.environ["ODDS_API_KEY"] = multi_odds._SENTINEL

    _INSTALLED = True
