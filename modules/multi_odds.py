from __future__ import annotations

"""Multi-source MLB odds adapter.

Keeps app_mlb.py compatible with its existing The-Odds-API-shaped payload while
adding Odds-API.io and SharpAPI as optional providers.  All providers are
normalized into the same event/bookmaker/market/outcome schema and a synthetic
"Multi-source consensus" bookmaker is prepended when enough data is available.

Secrets (environment or Streamlit secrets):
  ODDS_API_KEY       existing The Odds API key (optional)
  ODDS_API_IO_KEY    odds-api.io key (optional)
  SHARPAPI_KEY       SharpAPI key (optional)
  ODDS_API_IO_BOOKMAKERS optional comma-separated list, default Bet365,DraftKings
"""

import os
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

import requests

_SENTINEL = "__MULTI_ODDS_ONLY__"
_INSTALLED = False
_ORIGINAL_GET = None


def _secret(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        import streamlit as st
        value = str(st.secrets.get(name, "")).strip()
        if value:
            return value
    except Exception:
        pass
    return default


def alternate_provider_configured() -> bool:
    return bool(_secret("ODDS_API_IO_KEY") or _secret("SHARPAPI_KEY"))


def ensure_app_odds_gate_open() -> None:
    """app_mlb only enters its odds block when ODDS_API_KEY is non-empty.

    When only an alternate provider is configured, set a harmless sentinel.
    The requests bridge below recognizes it and never sends it to The Odds API.
    """
    if not os.getenv("ODDS_API_KEY", "").strip() and alternate_provider_configured():
        os.environ["ODDS_API_KEY"] = _SENTINEL


def _american_from_decimal(value: Any):
    try:
        dec = float(value)
        if dec <= 1.0:
            return None
        if dec >= 2.0:
            return round((dec - 1.0) * 100.0)
        return round(-100.0 / (dec - 1.0))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _american(value: Any, decimal: Any = None):
    try:
        if value not in (None, ""):
            v = float(value)
            if abs(v) >= 100:
                return round(v)
            if v > 1:
                return _american_from_decimal(v)
    except (TypeError, ValueError):
        pass
    return _american_from_decimal(decimal)


def _iso(value: Any):
    if value in (None, ""):
        return None
    try:
        # Preserve already-valid ISO strings. Matching code parses them with pandas.
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)):
            # tolerate seconds or milliseconds
            ts = float(value)
            if ts > 10_000_000_000:
                ts /= 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        pass
    return str(value)


def _event(home, away, commence, event_id, bookmakers):
    return {
        "id": str(event_id or f"{away}@{home}:{commence}"),
        "home_team": str(home or ""),
        "away_team": str(away or ""),
        "commence_time": _iso(commence),
        "bookmakers": bookmakers or [],
    }


def _market(key, outcomes):
    return {"key": key, "outcomes": outcomes}


def _outcome(name, price, point=None):
    out = {"name": str(name), "price": price}
    if point is not None:
        out["point"] = float(point)
    return out


def _fetch_odds_api_io(get_fn) -> list[dict]:
    key = _secret("ODDS_API_IO_KEY")
    if not key:
        return []
    base = "https://api.odds-api.io/v3"
    books = _secret("ODDS_API_IO_BOOKMAKERS", "Bet365,DraftKings")
    try:
        r = get_fn(f"{base}/events", params={"apiKey": key, "sport": "mlb"}, timeout=8)
        if getattr(r, "status_code", 0) != 200:
            return []
        raw_events = r.json()
        if isinstance(raw_events, dict):
            raw_events = raw_events.get("data") or raw_events.get("events") or []
        results = []
        for ev in raw_events or []:
            eid = ev.get("id") or ev.get("eventId")
            home = ev.get("home") or ev.get("home_team") or ev.get("homeTeam")
            away = ev.get("away") or ev.get("away_team") or ev.get("awayTeam")
            commence = ev.get("date") or ev.get("start_time") or ev.get("commence_time")
            if not eid or not home or not away:
                continue
            ro = get_fn(
                f"{base}/odds",
                params={"apiKey": key, "eventId": eid, "bookmakers": books},
                timeout=8,
            )
            if getattr(ro, "status_code", 0) != 200:
                continue
            payload = ro.json()
            bookmaker_map = payload.get("bookmakers", {}) if isinstance(payload, dict) else {}
            if not bookmaker_map and isinstance(payload, list):
                # Some responses wrap a single event in a list.
                candidate = payload[0] if payload else {}
                bookmaker_map = candidate.get("bookmakers", {}) if isinstance(candidate, dict) else {}
            normalized_books = []
            for book_name, markets in (bookmaker_map or {}).items():
                nm = []
                for m in markets or []:
                    mname = str(m.get("name") or m.get("key") or "").lower()
                    odds_rows = m.get("odds") or m.get("outcomes") or []
                    if isinstance(odds_rows, dict):
                        odds_rows = [odds_rows]
                    if mname in {"ml", "moneyline", "h2h"}:
                        outs = []
                        for row in odds_rows:
                            hp = _american(row.get("home"), row.get("home_decimal"))
                            ap = _american(row.get("away"), row.get("away_decimal"))
                            if hp is not None:
                                outs.append(_outcome(home, hp))
                            if ap is not None:
                                outs.append(_outcome(away, ap))
                        if len(outs) == 2:
                            nm.append(_market("h2h", outs))
                    elif mname in {"total", "totals", "ou", "over/under"}:
                        for row in odds_rows:
                            point = row.get("hdp", row.get("line", row.get("point")))
                            over = _american(row.get("over"), row.get("over_decimal"))
                            under = _american(row.get("under"), row.get("under_decimal"))
                            if point is not None and over is not None and under is not None:
                                nm.append(_market("totals", [_outcome("Over", over, point), _outcome("Under", under, point)]))
                                break
                    elif mname in {"spread", "spreads", "run line", "runline"}:
                        for row in odds_rows:
                            point = row.get("hdp", row.get("line", row.get("point")))
                            hp = _american(row.get("home"), row.get("home_decimal"))
                            ap = _american(row.get("away"), row.get("away_decimal"))
                            if point is not None and hp is not None and ap is not None:
                                p = float(point)
                                nm.append(_market("spreads", [_outcome(home, hp, p), _outcome(away, ap, -p)]))
                                break
                if nm:
                    normalized_books.append({"key": str(book_name).lower(), "title": str(book_name), "markets": nm})
            if normalized_books:
                results.append(_event(home, away, commence, eid, normalized_books))
        return results
    except Exception:
        return []


def _fetch_sharpapi(get_fn) -> list[dict]:
    key = _secret("SHARPAPI_KEY")
    if not key:
        return []
    url = "https://api.sharpapi.io/api/v1/odds"
    headers = {"Authorization": f"Bearer {key}", "X-API-Key": key, "Accept": "application/json"}
    try:
        r = get_fn(url, params={"league": "MLB"}, headers=headers, timeout=8)
        if getattr(r, "status_code", 0) != 200:
            return []
        payload = r.json()
        rows = payload.get("data", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            return []

        grouped = defaultdict(list)
        meta = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            home = row.get("home_team") or row.get("homeTeam")
            away = row.get("away_team") or row.get("awayTeam")
            eid = row.get("event_id") or row.get("eventId") or f"{away}@{home}:{row.get('start_time')}"
            if not home or not away:
                continue
            grouped[str(eid)].append(row)
            meta[str(eid)] = (home, away, row.get("start_time") or row.get("commence_time"))

        events = []
        for eid, event_rows in grouped.items():
            home, away, commence = meta[eid]
            by_book = defaultdict(list)
            for row in event_rows:
                by_book[str(row.get("sportsbook") or "SharpAPI")].append(row)
            books = []
            for book, book_rows in by_book.items():
                markets = []
                ml = {}
                totals = defaultdict(dict)
                spreads = defaultdict(dict)
                for row in book_rows:
                    market = str(row.get("market_type") or row.get("market") or "").lower()
                    selection = str(row.get("selection") or "")
                    stype = str(row.get("selection_type") or "").lower()
                    price = _american(row.get("odds_american"), row.get("odds_decimal"))
                    line = row.get("line")
                    if price is None:
                        continue
                    if market in {"moneyline", "ml", "h2h"}:
                        target = home if stype == "home" else away if stype == "away" else selection
                        if target:
                            ml[str(target)] = price
                    elif market in {"total_points", "total", "totals", "over_under"} and line is not None:
                        side = "Over" if stype == "over" or selection.lower().startswith("over") else "Under" if stype == "under" or selection.lower().startswith("under") else None
                        if side:
                            totals[float(line)][side] = price
                    elif market in {"point_spread", "spread", "spreads", "runline", "run_line"} and line is not None:
                        target = home if stype == "home" else away if stype == "away" else selection
                        if target:
                            spreads[str(target)]["point"] = float(line)
                            spreads[str(target)]["price"] = price
                if home in ml and away in ml:
                    markets.append(_market("h2h", [_outcome(home, ml[home]), _outcome(away, ml[away])]))
                for point, sides in totals.items():
                    if "Over" in sides and "Under" in sides:
                        markets.append(_market("totals", [_outcome("Over", sides["Over"], point), _outcome("Under", sides["Under"], point)]))
                        break
                if home in spreads and away in spreads:
                    hp, ap = spreads[home], spreads[away]
                    if abs(float(hp["point"]) + float(ap["point"])) < 1e-6:
                        markets.append(_market("spreads", [_outcome(home, hp["price"], hp["point"]), _outcome(away, ap["price"], ap["point"])]))
                if markets:
                    books.append({"key": book.lower(), "title": book, "markets": markets})
            if books:
                events.append(_event(home, away, commence, eid, books))
        return events
    except Exception:
        return []


def _match_key(ev: dict):
    def norm(s):
        return " ".join(str(s or "").lower().replace(".", "").split())
    return norm(ev.get("away_team")), norm(ev.get("home_team"))


def _merge_events(*event_lists: list[dict]) -> list[dict]:
    merged = {}
    for events in event_lists:
        for ev in events or []:
            key = _match_key(ev)
            if not all(key):
                continue
            if key not in merged:
                merged[key] = _event(ev.get("home_team"), ev.get("away_team"), ev.get("commence_time"), ev.get("id"), [])
            target = merged[key]
            if not target.get("commence_time") and ev.get("commence_time"):
                target["commence_time"] = ev.get("commence_time")
            for book in ev.get("bookmakers", []) or []:
                clone = dict(book)
                clone["title"] = str(book.get("title") or book.get("key") or "Book")
                target["bookmakers"].append(clone)
    for ev in merged.values():
        consensus = _consensus_book(ev)
        if consensus:
            ev["bookmakers"].insert(0, consensus)
    return list(merged.values())


def _median_price(values):
    vals = [float(v) for v in values if v is not None]
    return round(statistics.median(vals)) if vals else None


def _consensus_book(ev: dict):
    home, away = ev.get("home_team"), ev.get("away_team")
    ml_home, ml_away = [], []
    totals = defaultdict(lambda: {"Over": [], "Under": []})
    spreads = defaultdict(lambda: {"home": [], "away": []})

    for book in ev.get("bookmakers", []) or []:
        for market in book.get("markets", []) or []:
            key = market.get("key")
            outs = market.get("outcomes", []) or []
            if key == "h2h":
                for out in outs:
                    if str(out.get("name")) == str(home):
                        ml_home.append(out.get("price"))
                    elif str(out.get("name")) == str(away):
                        ml_away.append(out.get("price"))
            elif key == "totals":
                for out in outs:
                    if out.get("point") is None:
                        continue
                    point = float(out.get("point"))
                    side = str(out.get("name"))
                    if side in ("Over", "Under"):
                        totals[point][side].append(out.get("price"))
            elif key == "spreads":
                home_point = away_point = None
                hp = ap = None
                for out in outs:
                    if str(out.get("name")) == str(home):
                        home_point, hp = out.get("point"), out.get("price")
                    elif str(out.get("name")) == str(away):
                        away_point, ap = out.get("point"), out.get("price")
                if home_point is not None and away_point is not None and abs(float(home_point) + float(away_point)) < 1e-6:
                    spreads[float(home_point)]["home"].append(hp)
                    spreads[float(home_point)]["away"].append(ap)

    markets = []
    if ml_home and ml_away:
        markets.append(_market("h2h", [_outcome(home, _median_price(ml_home)), _outcome(away, _median_price(ml_away))]))

    valid_totals = {p: d for p, d in totals.items() if d["Over"] and d["Under"]}
    if valid_totals:
        # Prefer the line supported by the most books, then closest to median line.
        counts = {p: len(d["Over"]) + len(d["Under"]) for p, d in valid_totals.items()}
        max_count = max(counts.values())
        candidates = [p for p, c in counts.items() if c == max_count]
        point = statistics.median(candidates)
        point = min(candidates, key=lambda p: abs(p - point))
        d = valid_totals[point]
        markets.append(_market("totals", [_outcome("Over", _median_price(d["Over"]), point), _outcome("Under", _median_price(d["Under"]), point)]))

    valid_spreads = {p: d for p, d in spreads.items() if d["home"] and d["away"]}
    if valid_spreads:
        point = max(valid_spreads, key=lambda p: len(valid_spreads[p]["home"]) + len(valid_spreads[p]["away"]))
        d = valid_spreads[point]
        markets.append(_market("spreads", [_outcome(home, _median_price(d["home"]), point), _outcome(away, _median_price(d["away"]), -point)]))

    if not markets:
        return None
    return {"key": "multi_source_consensus", "title": "Multi-source consensus", "markets": markets}


class _SyntheticResponse:
    def __init__(self, data, status_code=200, headers=None):
        self._data = data
        self.status_code = status_code
        self.headers = headers or {}
        self.ok = 200 <= status_code < 300
        self.text = ""

    def json(self):
        return self._data


def _is_the_odds_api_call(url: Any) -> bool:
    return "api.the-odds-api.com/v4/sports/baseball_mlb/odds" in str(url)


def install_requests_bridge() -> None:
    """Intercept only the existing MLB odds request and enrich it safely."""
    global _INSTALLED, _ORIGINAL_GET
    if _INSTALLED:
        return
    ensure_app_odds_gate_open()
    _ORIGINAL_GET = requests.get

    def bridged_get(url, *args, **kwargs):
        if not _is_the_odds_api_call(url):
            return _ORIGINAL_GET(url, *args, **kwargs)

        primary_events = []
        primary_status = None
        # Never send our sentinel as a real API key.
        if _SENTINEL not in str(url):
            try:
                primary = _ORIGINAL_GET(url, *args, **kwargs)
                primary_status = getattr(primary, "status_code", None)
                if primary_status == 200:
                    data = primary.json()
                    primary_events = data if isinstance(data, list) else []
            except Exception:
                primary_status = None

        alt_a = _fetch_odds_api_io(_ORIGINAL_GET)
        alt_b = _fetch_sharpapi(_ORIGINAL_GET)
        merged = _merge_events(primary_events, alt_a, alt_b)
        if merged:
            return _SyntheticResponse(merged, 200)
        if primary_status is not None and _SENTINEL not in str(url):
            try:
                return primary
            except Exception:
                pass
        return _SyntheticResponse([], primary_status or 503)

    requests.get = bridged_get
    _INSTALLED = True
