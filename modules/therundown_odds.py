from __future__ import annotations

"""TheRundown V2 MLB odds provider normalized to MLB-Pred's odds schema."""

import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from . import multi_odds

log = logging.getLogger(__name__)

_INSTALLED = False
_MLB_SPORT_ID = 3
_CENTRAL = ZoneInfo("America/Chicago")
_CACHE_TTL_SECONDS = 300
_CACHE = {"date": None, "at": 0.0, "events": []}

_AFFILIATE_NAMES = {
    "2": "Bovada", "3": "Pinnacle", "4": "Sportsbetting", "6": "BetOnline",
    "11": "LowVig", "12": "Bodog", "14": "Intertops", "16": "Matchbook",
    "18": "YouWager", "19": "DraftKings", "21": "Unibet", "22": "BetMGM",
    "23": "FanDuel", "24": "theScore Bet", "25": "Kalshi", "26": "Polymarket",
}


def _secret(name: str, default: str = "") -> str:
    return multi_odds._secret(name, default)


def _norm(v) -> str:
    return " ".join(str(v or "").lower().replace(".", "").split())


def _float(v):
    try:
        return None if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return None


def _price(v):
    try:
        x = float(v)
        if abs(x - 0.0001) < 1e-9 or abs(x) < 100:
            return None
        return round(x)
    except (TypeError, ValueError):
        return None


def _team_name(t: dict) -> str:
    name = str(t.get("name") or t.get("team_name") or "").strip()
    mascot = str(t.get("mascot") or t.get("nickname") or "").strip()
    if name and mascot and mascot.lower() not in name.lower():
        return f"{name} {mascot}"
    return name or mascot


def _same_team(a: str, b: str) -> bool:
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return False
    if a == b:
        return True
    aa, bb = a.split(), b.split()
    return bool(aa and bb and aa[-1] == bb[-1] and len(aa[-1]) >= 3)


def _event_teams(event: dict):
    teams = [x for x in (event.get("teams") or []) if isinstance(x, dict)]
    if len(teams) < 2:
        return None
    away_obj = next((x for x in teams if x.get("is_away") is True), teams[0])
    home_obj = next((x for x in teams if x.get("is_home") is True), None)
    if home_obj is None:
        home_obj = next((x for x in teams if x is not away_obj and x.get("is_away") is False), None)
    if home_obj is None:
        home_obj = next((x for x in teams if x is not away_obj), teams[1])
    home, away = _team_name(home_obj), _team_name(away_obj)
    return (home_obj, away_obj, home, away) if home and away else None


def _selection(participant: dict, home_obj: dict, away_obj: dict, home: str, away: str):
    pname = str(participant.get("name") or participant.get("team_name") or "").strip()
    pid = str(participant.get("id") or participant.get("team_id") or "")
    hid = str(home_obj.get("id") or home_obj.get("team_id") or "")
    aid = str(away_obj.get("id") or away_obj.get("team_id") or "")
    if pid and hid and pid == hid:
        return home
    if pid and aid and pid == aid:
        return away
    if _same_team(pname, home):
        return home
    if _same_team(pname, away):
        return away
    return pname


def _market_kind(m: dict):
    try:
        mid = int(m.get("market_id"))
    except (TypeError, ValueError):
        mid = 0
    if mid == 1:
        return "ml"
    if mid == 2:
        return "spread"
    if mid == 3:
        return "total"
    name = _norm(m.get("name") or m.get("market_description"))
    if "moneyline" in name:
        return "ml"
    if "handicap" in name or "spread" in name or "run line" in name:
        return "spread"
    if "total" in name or "over/under" in name:
        return "total"
    return None


def _line_prices(line: dict):
    prices = line.get("prices") or {}
    if isinstance(prices, dict):
        return [(str(k), v) for k, v in prices.items()]
    if isinstance(prices, list):
        out = []
        for row in prices:
            if isinstance(row, dict):
                aid = row.get("affiliate_id") or row.get("affiliate") or row.get("sportsbook_id")
                if aid is not None:
                    out.append((str(aid), row))
        return out
    return []


def _has_core_markets(events) -> bool:
    """A filtered response can contain events but omit markets entirely."""
    if not isinstance(events, list):
        return False
    for event in events:
        if not isinstance(event, dict):
            continue
        for market in event.get("markets") or []:
            if isinstance(market, dict) and _market_kind(market) in {"ml", "spread", "total"}:
                return True
    return False


def _request_payload(get_fn, url: str, headers: dict, affiliate_ids: str):
    narrow = {
        "market_ids": "1,2,3",
        "affiliate_ids": affiliate_ids,
        "main_line": "true",
        "hide_closed": "true",
        "offset": "300",
    }
    response = get_fn(url, params=narrow, headers=headers, timeout=12)
    if getattr(response, "status_code", 0) == 200:
        payload = response.json()
        events = payload.get("events", []) if isinstance(payload, dict) else []
        if events and _has_core_markets(events):
            log.warning("TheRundown narrow response usable: %d events with markets", len(events))
            return payload
        if events:
            log.warning("TheRundown narrow response has %d events but no core markets; using raw fallback", len(events))
    else:
        log.warning("TheRundown narrow request failed: HTTP %s", getattr(response, "status_code", None))

    # Verified against this account: the unfiltered endpoint returns markets/participants/lines/prices.
    response = get_fn(url, headers=headers, timeout=12)
    if getattr(response, "status_code", 0) != 200:
        log.warning("TheRundown fallback request failed: HTTP %s", getattr(response, "status_code", None))
        return None
    payload = response.json()
    events = payload.get("events", []) if isinstance(payload, dict) else []
    log.warning("TheRundown raw fallback returned %d events", len(events) if isinstance(events, list) else 0)
    return payload


def _fetch_therundown(get_fn) -> list[dict]:
    key = _secret("THERUNDOWN_KEY")
    if not key:
        return []

    slate_date = datetime.now(timezone.utc).astimezone(_CENTRAL).date().isoformat()
    now = time.monotonic()
    if _CACHE.get("date") == slate_date and now - float(_CACHE.get("at") or 0) < _CACHE_TTL_SECONDS:
        return list(_CACHE.get("events") or [])

    affiliate_ids = _secret("THERUNDOWN_AFFILIATE_IDS", "19,22,23")
    preferred = {x.strip() for x in affiliate_ids.split(",") if x.strip()}
    url = f"https://therundown.io/api/v2/sports/{_MLB_SPORT_ID}/events/{slate_date}"
    headers = {"X-TheRundown-Key": key, "Accept": "application/json"}

    try:
        payload = _request_payload(get_fn, url, headers, affiliate_ids)
        events = payload.get("events", []) if isinstance(payload, dict) else []
        if not isinstance(events, list):
            return []

        normalized = []
        market_rows = participant_rows = valid_price_rows = 0

        for event in events:
            if not isinstance(event, dict):
                continue
            ti = _event_teams(event)
            if not ti:
                continue
            home_obj, away_obj, home, away = ti
            by_book = defaultdict(lambda: {"ml": {}, "totals": defaultdict(dict), "spreads": defaultdict(dict)})

            for market in event.get("markets") or []:
                if not isinstance(market, dict):
                    continue
                kind = _market_kind(market)
                if kind is None:
                    continue
                # The observed payload uses period_id=0 for full-game MLB markets.
                if market.get("period_id") not in (None, "", 0, "0"):
                    continue
                market_rows += 1

                for participant in market.get("participants") or []:
                    if not isinstance(participant, dict):
                        continue
                    participant_rows += 1
                    pname = str(participant.get("name") or "").strip()
                    ptype = _norm(participant.get("type"))
                    selection = _selection(participant, home_obj, away_obj, home, away)
                    lines = participant.get("lines") or []
                    if isinstance(lines, dict):
                        lines = [lines]

                    for line in lines:
                        if not isinstance(line, dict):
                            continue
                        point = _float(line.get("value"))
                        for book_id, pobj in _line_prices(line):
                            if preferred and book_id not in preferred:
                                continue
                            if not isinstance(pobj, dict):
                                pobj = {"price": pobj}
                            if pobj.get("is_main_line") is False:
                                continue
                            price = _price(pobj.get("price", pobj.get("odds")))
                            if price is None:
                                continue
                            valid_price_rows += 1
                            slot = by_book[book_id]

                            if kind == "ml" and selection in (home, away):
                                slot["ml"][selection] = price
                            elif kind == "spread" and selection in (home, away) and point is not None:
                                slot["spreads"][point][selection] = price
                            elif kind == "total" and point is not None:
                                label = _norm(pname)
                                side = "Over" if ("over" in ptype or label.startswith("over")) else "Under" if ("under" in ptype or label.startswith("under")) else None
                                if side:
                                    slot["totals"][point][side] = price

            books = []
            for book_id, data in by_book.items():
                markets = []
                if home in data["ml"] and away in data["ml"]:
                    markets.append(multi_odds._market("h2h", [
                        multi_odds._outcome(home, data["ml"][home]),
                        multi_odds._outcome(away, data["ml"][away]),
                    ]))

                totals = [(p, s) for p, s in data["totals"].items() if "Over" in s and "Under" in s]
                if totals:
                    p, sides = totals[0]
                    markets.append(multi_odds._market("totals", [
                        multi_odds._outcome("Over", sides["Over"], p),
                        multi_odds._outcome("Under", sides["Under"], p),
                    ]))

                for p, sides in data["spreads"].items():
                    opposite = data["spreads"].get(-float(p), {})
                    if home in sides and away in opposite:
                        markets.append(multi_odds._market("spreads", [
                            multi_odds._outcome(home, sides[home], p),
                            multi_odds._outcome(away, opposite[away], -float(p)),
                        ]))
                        break

                if markets:
                    books.append({"key": f"therundown_{book_id}", "title": _AFFILIATE_NAMES.get(book_id, f"TheRundown {book_id}"), "markets": markets})

            if books:
                normalized.append(multi_odds._event(
                    home, away, event.get("event_date"),
                    event.get("event_id") or event.get("id"), books,
                ))

        _CACHE.update({"date": slate_date, "at": now, "events": normalized})
        log.warning(
            "TheRundown normalized %d of %d MLB events for %s (markets=%d participants=%d valid_prices=%d)",
            len(normalized), len(events), slate_date, market_rows, participant_rows, valid_price_rows,
        )
        return normalized
    except Exception:
        log.exception("TheRundown MLB odds normalization failed")
        return []


def install_therundown_provider() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    original_sharp = multi_odds._fetch_sharpapi

    def fetch_alternates(get_fn):
        return list(_fetch_therundown(get_fn)) + list(original_sharp(get_fn))

    multi_odds._fetch_sharpapi = fetch_alternates
    if _secret("THERUNDOWN_KEY") and not os.getenv("ODDS_API_KEY", "").strip():
        os.environ["ODDS_API_KEY"] = multi_odds._SENTINEL
    _INSTALLED = True
