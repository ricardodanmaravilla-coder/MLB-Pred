from __future__ import annotations

"""TheRundown V2 MLB odds provider for the existing multi-odds bridge.

This module keeps the rest of MLB-Pred on its existing The-Odds-API-shaped
contract while normalizing TheRundown V2 event/market/participant/line/price
rows into h2h, spreads and totals.
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
_MLB_SPORT_ID = 3  # confirmed by this account's /api/v2/sports catalog
_CENTRAL = ZoneInfo("America/Chicago")
_CACHE_TTL_SECONDS = 300
_CACHE: dict[str, object] = {"date": None, "at": 0.0, "events": []}

_AFFILIATE_NAMES = {
    "2": "Bovada", "3": "Pinnacle", "4": "Sportsbetting", "6": "BetOnline",
    "11": "LowVig", "12": "Bodog", "14": "Intertops", "16": "Matchbook",
    "18": "YouWager", "19": "DraftKings", "21": "Unibet", "22": "BetMGM",
    "23": "FanDuel", "24": "theScore Bet", "25": "Kalshi", "26": "Polymarket",
}


def _secret(name: str, default: str = "") -> str:
    return multi_odds._secret(name, default)


def _price(value):
    try:
        v = float(value)
        if abs(v - 0.0001) < 1e-9:
            return None
        # American odds used by MLB-Pred. Even money is valid.
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
    na, nb = _norm_name(a), _norm_name(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    ta, tb = na.split(), nb.split()
    return bool(ta and tb and len(ta[-1]) >= 4 and ta[-1] == tb[-1])


def _team_name(team: dict) -> str:
    # V2 payloads may provide a full name, nickname/mascot, or normalized names.
    full = str(team.get("name") or team.get("team_name") or "").strip()
    mascot = str(team.get("mascot") or team.get("nickname") or "").strip()
    if full and mascot and mascot.lower() not in full.lower():
        return f"{full} {mascot}".strip()
    return full or mascot


def _event_teams(event: dict):
    teams = [t for t in (event.get("teams") or event.get("teams_normalized") or []) if isinstance(t, dict)]
    if len(teams) < 2:
        return None

    away_obj = next((t for t in teams if t.get("is_away") is True), None)
    home_obj = next((t for t in teams if t.get("is_home") is True), None)

    # Some V2 payloads expose only is_away. Official examples use away first.
    if away_obj is None:
        away_obj = teams[0]
    if home_obj is None:
        home_obj = next((t for t in teams if t is not away_obj and t.get("is_away") is False), None)
    if home_obj is None:
        home_obj = teams[1] if teams[1] is not away_obj else teams[0]

    away = _team_name(away_obj)
    home = _team_name(home_obj)
    if not home or not away:
        return None
    return home_obj, away_obj, home, away


def _team_selection(participant: dict, home_obj: dict, away_obj: dict, home: str, away: str):
    pid = str(participant.get("id") or participant.get("team_id") or participant.get("participant_id") or "")
    home_id = str(home_obj.get("id") or home_obj.get("team_id") or home_obj.get("participant_id") or "")
    away_id = str(away_obj.get("id") or away_obj.get("team_id") or away_obj.get("participant_id") or "")
    pname = str(participant.get("name") or participant.get("team_name") or "").strip()

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
        if isinstance(events, list) and events:
            return payload
    else:
        log.warning("TheRundown narrow request failed: HTTP %s", getattr(response, "status_code", None))

    # This account has been verified to return MLB rows on the unfiltered route.
    response = get_fn(url, headers=headers, timeout=12)
    if getattr(response, "status_code", 0) != 200:
        log.warning("TheRundown fallback request failed: HTTP %s", getattr(response, "status_code", None))
        return None
    return response.json()


def _market_kind(market: dict):
    """Return one of ml/spread/total, tolerating catalog/schema variations."""
    raw_id = market.get("market_id", market.get("id"))
    try:
        mid = int(raw_id)
    except (TypeError, ValueError):
        mid = 0
    if mid == 1:
        return "ml"
    if mid == 2:
        return "spread"
    if mid == 3:
        return "total"

    name = _norm_name(market.get("name") or market.get("market_name") or market.get("type"))
    if "moneyline" in name or name in {"ml", "h2h"}:
        return "ml"
    if "spread" in name or "run line" in name or "runline" in name:
        return "spread"
    if "total" in name or "over under" in name:
        return "total"
    return None


def _full_game_market(market: dict) -> bool:
    """Exclude clearly non-full-game periods, but don't reject missing/variant IDs."""
    period_name = _norm_name(market.get("period_name") or market.get("period") or "")
    if period_name:
        bad = ("1st inning", "first inning", "2nd inning", "3rd inning", "4th inning",
               "5th inning", "6th inning", "7th inning", "8th inning", "9th inning",
               "first 5", "1st 5", "first five", "first half")
        if any(x in period_name for x in bad):
            return False
        if "full" in period_name or "game" in period_name:
            return True

    raw = market.get("period_id")
    # Do not assume full game is always period_id=0; catalog versions differ.
    if raw in (None, "", 0, "0", 1, "1"):
        return True
    return True


def _line_prices(line: dict):
    prices = line.get("prices") or line.get("affiliate_prices") or {}
    if isinstance(prices, dict):
        return prices.items()
    if isinstance(prices, list):
        out = []
        for row in prices:
            if not isinstance(row, dict):
                continue
            aid = row.get("affiliate_id") or row.get("affiliate") or row.get("sportsbook_id")
            if aid is not None:
                out.append((str(aid), row))
        return out
    return []


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
        market_rows = participant_rows = valid_price_rows = 0

        for event in events:
            if not isinstance(event, dict):
                continue
            team_info = _event_teams(event)
            if not team_info:
                continue
            home_obj, away_obj, home, away = team_info

            by_book = defaultdict(lambda: {
                "ml": {},
                "totals": defaultdict(dict),
                "spreads": defaultdict(dict),
            })

            for market in event.get("markets") or []:
                if not isinstance(market, dict):
                    continue
                kind = _market_kind(market)
                if kind is None or not _full_game_market(market):
                    continue
                market_rows += 1

                participants = market.get("participants") or market.get("outcomes") or []
                for participant in participants:
                    if not isinstance(participant, dict):
                        continue
                    participant_rows += 1
                    pname = str(participant.get("name") or participant.get("label") or "").strip()
                    ptype = _norm_name(participant.get("type") or participant.get("participant_type") or "")
                    selection = _team_selection(participant, home_obj, away_obj, home, away)

                    lines = participant.get("lines") or []
                    # Some feeds may put one line directly on the participant.
                    if isinstance(lines, dict):
                        lines = [lines]
                    if not lines and (participant.get("prices") or participant.get("price") is not None):
                        lines = [participant]

                    for line in lines:
                        if not isinstance(line, dict):
                            continue
                        point = _float(line.get("value", line.get("line", line.get("point"))))

                        for affiliate_id, price_obj in _line_prices(line):
                            aid = str(affiliate_id)
                            if allowed_affiliates and aid not in allowed_affiliates:
                                continue
                            if not isinstance(price_obj, dict):
                                price_obj = {"price": price_obj}

                            main_flag = price_obj.get("is_main_line", line.get("is_main_line"))
                            if main_flag is False:
                                continue
                            if price_obj.get("closed_at") not in (None, ""):
                                continue

                            price = _price(price_obj.get("price", price_obj.get("odds")))
                            if price is None:
                                continue
                            valid_price_rows += 1
                            slot = by_book[aid]

                            if kind == "ml" and selection in (home, away):
                                slot["ml"][selection] = price
                            elif kind == "spread" and selection in (home, away) and point is not None:
                                slot["spreads"][point][selection] = price
                            elif kind == "total" and point is not None:
                                side = None
                                label = _norm_name(pname)
                                if "over" in ptype or label.startswith("over"):
                                    side = "Over"
                                elif "under" in ptype or label.startswith("under"):
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

                valid_totals = [(p, s) for p, s in data["totals"].items() if "Over" in s and "Under" in s]
                if valid_totals:
                    point, sides = valid_totals[0]
                    markets.append(multi_odds._market("totals", [
                        multi_odds._outcome("Over", sides["Over"], point),
                        multi_odds._outcome("Under", sides["Under"], point),
                    ]))

                spread_market = None
                for point, selections in data["spreads"].items():
                    if home not in selections:
                        continue
                    away_rows = data["spreads"].get(-float(point), {})
                    if away in away_rows:
                        spread_market = multi_odds._market("spreads", [
                            multi_odds._outcome(home, selections[home], point),
                            multi_odds._outcome(away, away_rows[away], -float(point)),
                        ])
                        break
                if spread_market:
                    markets.append(spread_market)

                if markets:
                    books.append({
                        "key": f"therundown_{aid}",
                        "title": _AFFILIATE_NAMES.get(aid, f"TheRundown {aid}"),
                        "markets": markets,
                    })

            if books:
                normalized.append(multi_odds._event(
                    home, away, event.get("event_date"),
                    event.get("event_id") or event.get("event_uuid") or event.get("id"),
                    books,
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
