
import json
import math
import os
import random
import re
import sqlite3
import time
import html
from datetime import date, datetime, timedelta
from pathlib import Path
from collections import Counter
from html.parser import HTMLParser
from urllib.parse import quote_plus

import pandas as pd
import requests
import streamlit as st

# ============================================================
# MATCHLAB LIVE 10X
# Free public football data + live Internet research
# ============================================================

st.set_page_config(
    page_title="MATCHLAB LIVE 10X",
    page_icon="⚽",
    layout="wide",
)

DB_PATH = Path("matchlab_history.db")
TSDB_BASE = "https://www.thesportsdb.com/api/v1/json/123"

st.markdown(
    """
    <style>
    .title {font-size:42px;font-weight:800;margin-bottom:0;}
    .subtitle {color:#8f9aaa;margin-bottom:18px;}
    .welcome {padding:18px;border-radius:14px;background:rgba(80,100,130,.12);
              border:1px solid rgba(130,150,180,.20);margin-bottom:20px;}
    .reason {padding:14px;border-radius:12px;margin-bottom:10px;
             background:rgba(80,100,130,.08);border:1px solid rgba(130,150,180,.14);}
    .value-box {padding:16px;border-radius:14px;background:rgba(30,140,80,.10);
                border:1px solid rgba(30,160,90,.25);}
    .hero {padding:26px;border-radius:22px;background:linear-gradient(135deg,rgba(30,90,180,.18),rgba(40,180,130,.10));
            border:1px solid rgba(120,150,200,.22);margin:8px 0 22px;}
    .hero h1 {font-size:44px;letter-spacing:-1px;margin:0;}
    .hero p {margin:7px 0 0;color:#9aa7b8;font-size:15px;}
    .signal {padding:14px 16px;border-radius:16px;background:rgba(80,100,130,.08);border:1px solid rgba(130,150,180,.16);}
    .decision {padding:18px;border-radius:18px;background:rgba(30,140,80,.10);border:1px solid rgba(30,160,90,.25);font-size:18px;}
    .muted {color:#8f9aaa;font-size:13px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# GENERAL HELPERS
# ============================================================

def clamp(value, low, high):
    return max(low, min(high, value))

def norm(value):
    return " ".join(str(value or "").strip().lower().split())

def pct(value):
    return f"{value * 100:.1f}%"

def fair_odds(probability):
    if probability <= 0:
        return None
    return 1.0 / probability

def implied_probability(decimal_odds):
    if decimal_odds is None or decimal_odds <= 1:
        return None
    return 1.0 / decimal_odds

def safe_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default

def parse_date(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None

# ============================================================
# SQLITE HISTORY
# ============================================================

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                fixture_id TEXT,
                match_date TEXT,
                competition TEXT,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                home_probability REAL,
                draw_probability REAL,
                away_probability REAL,
                home_xg REAL,
                away_xg REAL,
                predicted_outcome TEXT,
                predicted_score TEXT,
                confidence_score REAL,
                model_source TEXT,
                actual_home_goals INTEGER,
                actual_away_goals INTEGER,
                result_checked INTEGER DEFAULT 0,
                value_summary TEXT,
                reasons_json TEXT,
                raw_json TEXT
            )
        """)
        conn.commit()

init_db()

def save_prediction(record):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO predictions (
                created_at, fixture_id, match_date, competition,
                home_team, away_team,
                home_probability, draw_probability, away_probability,
                home_xg, away_xg, predicted_outcome, predicted_score,
                confidence_score, model_source, value_summary,
                reasons_json, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            record.get("fixture_id"),
            record.get("match_date"),
            record.get("competition"),
            record["home_team"], record["away_team"],
            record["home_probability"], record["draw_probability"],
            record["away_probability"], record["home_xg"], record["away_xg"],
            record["predicted_outcome"], record["predicted_score"],
            record.get("confidence_score"), record.get("model_source"),
            record.get("value_summary", ""),
            json.dumps(record.get("reasons", [])),
            json.dumps(record, default=str),
        ))
        conn.commit()

def load_history(limit=100):
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("""
            SELECT id, created_at, fixture_id, match_date, competition,
                   home_team, away_team, home_probability, draw_probability,
                   away_probability, home_xg, away_xg, predicted_outcome,
                   predicted_score, confidence_score, model_source,
                   actual_home_goals, actual_away_goals, result_checked,
                   value_summary
            FROM predictions ORDER BY id DESC LIMIT ?
        """, conn, params=(limit,))

def delete_prediction(prediction_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM predictions WHERE id = ?", (int(prediction_id),))
        conn.commit()

def update_actual_result(prediction_id, home_goals, away_goals):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            UPDATE predictions
            SET actual_home_goals=?, actual_away_goals=?, result_checked=1
            WHERE id=?
        """, (home_goals, away_goals, int(prediction_id)))
        conn.commit()

# ============================================================
# FREE STRUCTURED FOOTBALL DATA — THE SPORTsDB V1
# ============================================================

class FreeFootballData:
    """Uses TheSportsDB's documented free V1 API (key 123)."""

    def get(self, endpoint, params=None, timeout=15):
        url = TSDB_BASE + endpoint
        response = requests.get(
            url,
            params=params or {},
            timeout=timeout,
            headers={"User-Agent": "MATCHLAB/4.0 football research"},
        )
        response.raise_for_status()
        payload = response.json()
        return payload

    def search_team(self, name):
        data = self.get("/searchteams.php", {"t": name})
        return data.get("teams") or []

    def search_event(self, title):
        data = self.get("/searchevents.php", {"e": title})
        return data.get("event") or data.get("events") or []

    def events_day(self, day):
        data = self.get("/eventsday.php", {"d": day, "s": "Soccer"})
        return data.get("events") or []

    def eventslast(self, team_id):
        data = self.get("/eventslast.php", {"id": team_id})
        return data.get("results") or data.get("events") or []

    def eventsnext(self, team_id):
        data = self.get("/eventsnext.php", {"id": team_id})
        return data.get("events") or data.get("results") or []

    def event(self, event_id):
        data = self.get("/lookupevent.php", {"id": event_id})
        events = data.get("events") or []
        return events[0] if events else None

football = FreeFootballData()

def team_name_matches(actual, requested):
    a, r = norm(actual), norm(requested)
    if a == r:
        return True
    a2 = re.sub(r"[^a-z0-9 ]", "", a)
    r2 = re.sub(r"[^a-z0-9 ]", "", r)
    return a2 == r2 or r2 in a2 or a2 in r2

def choose_team(candidates, requested):
    if not candidates:
        return None
    exact = [x for x in candidates if team_name_matches(x.get("strTeam"), requested)]
    return exact[0] if exact else candidates[0]

def event_matches_fixture(event, home_name, away_name):
    h = event.get("strHomeTeam", "")
    a = event.get("strAwayTeam", "")
    return team_name_matches(h, home_name) and team_name_matches(a, away_name)

def event_to_fixture(event):
    return {
        "fixture": {
            "id": event.get("idEvent"),
            "date": f"{event.get('dateEvent','')}T{event.get('strTime','00:00:00')}",
            "venue": {"name": event.get("strVenue") or "Unknown venue"},
            "status": {"short": "FT" if event.get("intHomeScore") is not None else "NS"},
        },
        "league": {"name": event.get("strLeague") or "Unknown competition"},
        "teams": {
            "home": {"id": event.get("idHomeTeam"), "name": event.get("strHomeTeam")},
            "away": {"id": event.get("idAwayTeam"), "name": event.get("strAwayTeam")},
        },
        "goals": {
            "home": safe_float(event.get("intHomeScore")),
            "away": safe_float(event.get("intAwayScore")),
        },
        "raw_event": event,
    }

def _extract_iso_dates(text_value):
    """Extract common ISO, English, and Spanish calendar dates from web text."""
    s = str(text_value or "")
    found = []

    # ISO / slash forms.
    for y, m, d in re.findall(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", s):
        try:
            found.append(date(int(y), int(m), int(d)))
        except ValueError:
            pass

    month_map = {
        "jan": 1, "january": 1, "ene": 1, "enero": 1,
        "feb": 2, "february": 2, "febrero": 2,
        "mar": 3, "march": 3, "marzo": 3,
        "apr": 4, "april": 4, "abr": 4, "abril": 4,
        "may": 5, "mayo": 5,
        "jun": 6, "june": 6, "junio": 6,
        "jul": 7, "july": 7, "julio": 7,
        "aug": 8, "august": 8, "ago": 8, "agosto": 8,
        "sep": 9, "sept": 9, "september": 9, "septiembre": 9,
        "oct": 10, "october": 10, "octubre": 10,
        "nov": 11, "november": 11, "noviembre": 11,
        "dec": 12, "december": 12, "dic": 12, "diciembre": 12,
    }

    # "October 21, 2026" / "Oct 21 2026"
    for mon, day_num, year_num in re.findall(
        r"\b([A-Za-zÁÉÍÓÚáéíóú]{3,12})\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*|\s+)(20\d{2})\b",
        s,
    ):
        m = month_map.get(mon.lower())
        if m:
            try:
                found.append(date(int(year_num), m, int(day_num)))
            except ValueError:
                pass

    # "21 October 2026" / "21 Oct 2026"
    for day_num, mon, year_num in re.findall(
        r"\b(\d{1,2})\s+([A-Za-zÁÉÍÓÚáéíóú]{3,12})\s+(20\d{2})\b",
        s,
    ):
        m = month_map.get(mon.lower())
        if m:
            try:
                found.append(date(int(year_num), m, int(day_num)))
            except ValueError:
                pass

    # Spanish: "21 de octubre de 2026"
    for day_num, mon, year_num in re.findall(
        r"\b(\d{1,2})\s+de\s+([A-Za-zÁÉÍÓÚáéíóú]{3,12})\s+de\s+(20\d{2})\b",
        s,
        flags=re.IGNORECASE,
    ):
        m = month_map.get(mon.lower())
        if m:
            try:
                found.append(date(int(year_num), m, int(day_num)))
            except ValueError:
                pass

    # Remove duplicates while preserving order.
    unique = []
    seen = set()
    for d in found:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return unique

def _web_fixture_candidates(home_name, away_name, match_date, competition_hint=""):
    """Use public web/news search as a genuine fixture-discovery layer."""
    queries = [
        f'"{home_name}" "{away_name}" football {match_date.year}',
        f'"{home_name}" "{away_name}" fixture {competition_hint}'.strip(),
        f'"{home_name}" vs "{away_name}" {competition_hint}'.strip(),
    ]
    results = []
    seen = set()

    for q in queries:
        try:
            found = public_web_search(q, max_results=8)
        except Exception:
            found = []
        for r in found:
            key = r.get("url") or r.get("title")
            if key and key not in seen:
                seen.add(key)
                results.append(r)

    candidates = []
    hn = norm(home_name)
    an = norm(away_name)

    for r in results:
        text = f"{r.get('title','')} {r.get('content','')}"
        blob = norm(text)

        # Search engines often abbreviate club names (e.g. "Man Utd") or
        # omit punctuation. Accept a strong token overlap instead of requiring
        # the entire normalized name to appear verbatim.
        def name_present(name):
            n = norm(name)
            if n in blob:
                return True
            tokens = [t for t in re.findall(r"[a-z0-9]+", n) if len(t) >= 3]
            if not tokens:
                return False
            hits = sum(1 for t in tokens if t in blob)
            return hits >= max(1, min(2, len(tokens)))

        if not name_present(home_name) or not name_present(away_name):
            continue

        dates_found = _extract_iso_dates(text)
        candidates.append({
            "result": r,
            "dates": dates_found,
        })

    return candidates


def find_fixture(home_name, away_name, match_date, competition_hint=""):
    wanted_date = match_date.isoformat()

    # ---------- 1. Free structured football source ----------
    titles = [
        f"{home_name} vs {away_name}",
        f"{home_name} v {away_name}",
        f"{home_name} {away_name}",
    ]
    for title in titles:
        try:
            events = football.search_event(title)
            for event in events:
                if (
                    str(event.get("dateEvent", ""))[:10] == wanted_date
                    and event_matches_fixture(event, home_name, away_name)
                ):
                    fixture = event_to_fixture(event)
                    return fixture, fixture["teams"]["home"], fixture["teams"]["away"]
        except Exception:
            continue

    try:
        events = football.events_day(wanted_date)
        for event in events:
            if event_matches_fixture(event, home_name, away_name):
                fixture = event_to_fixture(event)
                return fixture, fixture["teams"]["home"], fixture["teams"]["away"]
    except Exception:
        pass

    # ---------- 1B. Team-schedule fallback ----------
    # Some free-data searches miss a fixture even though it is present in
    # the team's upcoming schedule. Search both teams' schedules before
    # relying on HTML/news search. This is more reliable for ordinary
    # fixtures and does not require a paid API key.
    try:
        home_candidates = football.search_team(home_name)
        away_candidates = football.search_team(away_name)
        home_match = choose_team(home_candidates, home_name)
        away_match = choose_team(away_candidates, away_name)
        home_id = home_match.get("idTeam") if home_match else None
        away_id = away_match.get("idTeam") if away_match else None

        schedule_events = []
        for team_id in (home_id, away_id):
            if not team_id:
                continue
            try:
                schedule_events.extend(football.eventsnext(team_id))
            except Exception:
                continue

        seen_events = set()
        for event in schedule_events:
            eid = str(event.get("idEvent") or "")
            if eid and eid in seen_events:
                continue
            if eid:
                seen_events.add(eid)
            event_date = str(event.get("dateEvent") or "")[:10]
            if event_date == wanted_date and event_matches_fixture(event, home_name, away_name):
                fixture = event_to_fixture(event)
                return fixture, fixture["teams"]["home"], fixture["teams"]["away"]
    except Exception:
        pass

    # ---------- 2. Public Internet fixture discovery ----------
    try:
        candidates = _web_fixture_candidates(
            home_name, away_name, match_date, competition_hint
        )

        # Prefer an exact date, then a one-day timezone/display difference.
        best = None
        best_distance = 999

        for c in candidates:
            for d in c["dates"]:
                distance = abs((d - match_date).days)
                if distance < best_distance:
                    best_distance = distance
                    best = (c, d)

        if best:
            chosen, verified_date = best
            best_distance = abs((verified_date - match_date).days)

            accepted = best_distance <= 1
            if best_distance == 2:
                corroborating = 0
                for other in candidates:
                    if other is chosen:
                        continue
                    if any(abs((d - verified_date).days) <= 1 for d in other["dates"]):
                        corroborating += 1
                accepted = corroborating >= 1

            if accepted:
                chosen_result = chosen["result"]
                home_candidates = football.search_team(home_name)
                away_candidates = football.search_team(away_name)
                home_match = choose_team(home_candidates, home_name)
                away_match = choose_team(away_candidates, away_name)

                home_id = home_match.get("idTeam") if home_match else None
                away_id = away_match.get("idTeam") if away_match else None

                fixture = {
                    "fixture": {
                        "id": f"WEB-{verified_date.isoformat()}-{norm(home_name)}-{norm(away_name)}",
                        "date": f"{verified_date.isoformat()}T00:00:00",
                        "venue": {"name": "Web-researched venue / verify source"},
                        "status": {"short": "NS"},
                    },
                    "league": {
                        "name": competition_hint or "Web-researched competition"
                    },
                    "teams": {
                        "home": {"id": home_id, "name": home_name},
                        "away": {"id": away_id, "name": away_name},
                    },
                    "goals": {"home": None, "away": None},
                    "raw_event": {
                        "web_verified": True,
                        "requested_date": wanted_date,
                        "verified_date": verified_date.isoformat(),
                        "date_difference_days": best_distance,
                        "fixture_source": chosen_result,
                        "research_results": [x["result"] for x in candidates],
                    },
                }
                return fixture, fixture["teams"]["home"], fixture["teams"]["away"]

    except Exception:
        pass

    # ---------- 3. Last-resort user-supplied fixture fallback ----------
    # Do not block the prediction engine just because a free provider/search
    # engine is temporarily unavailable. The user has explicitly supplied the
    # two teams and date, so MATCHLAB can proceed as a PROVISIONAL fixture and
    # clearly label it. Current team news and any available form data are still
    # collected separately below. This keeps the prediction engine usable even
    # when free fixture databases rate-limit or search pages change.
    try:
        home_candidates = football.search_team(home_name)
    except Exception:
        home_candidates = []
    try:
        away_candidates = football.search_team(away_name)
    except Exception:
        away_candidates = []

    home_match = choose_team(home_candidates, home_name)
    away_match = choose_team(away_candidates, away_name)

    home_id = home_match.get("idTeam") if home_match else None
    away_id = away_match.get("idTeam") if away_match else None
    canonical_home = home_match.get("strTeam") if home_match else home_name
    canonical_away = away_match.get("strTeam") if away_match else away_name

    fixture = {
        "fixture": {
            "id": f"PROVISIONAL-{wanted_date}-{norm(home_name)}-{norm(away_name)}",
            "date": f"{wanted_date}T00:00:00",
            "venue": {"name": "Venue not independently verified"},
            "status": {"short": "NS"},
        },
        "league": {"name": competition_hint or "Competition not independently verified"},
        "teams": {
            "home": {"id": home_id, "name": canonical_home},
            "away": {"id": away_id, "name": canonical_away},
        },
        "goals": {"home": None, "away": None},
        "raw_event": {
            "provisional": True,
            "requested_date": wanted_date,
            "fixture_source": "User-supplied fixture; external verification unavailable",
            "verification_note": (
                "Free fixture sources did not return a matching event. "
                "MATCHLAB proceeded without inventing a different date or result."
            ),
        },
    }
    return fixture, fixture["teams"]["home"], fixture["teams"]["away"]

def event_to_row(event, team_id):
    if not event:
        return None
    hg = safe_float(event.get("intHomeScore"))
    ag = safe_float(event.get("intAwayScore"))
    if hg is None or ag is None:
        return None

    home_id = str(event.get("idHomeTeam", ""))
    away_id = str(event.get("idAwayTeam", ""))
    tid = str(team_id)

    if home_id == tid:
        return {
            "date": event.get("dateEvent", ""),
            "team_id": team_id,
            "opponent_id": event.get("idAwayTeam"),
            "venue": "home",
            "gf": hg, "ga": ag,
            "result": "W" if hg > ag else "D" if hg == ag else "L",
            "competition": event.get("strLeague", ""),
        }
    if away_id == tid:
        return {
            "date": event.get("dateEvent", ""),
            "team_id": team_id,
            "opponent_id": event.get("idHomeTeam"),
            "venue": "away",
            "gf": ag, "ga": hg,
            "result": "W" if ag > hg else "D" if ag == hg else "L",
            "competition": event.get("strLeague", ""),
        }
    return None

@st.cache_data(ttl=900, show_spinner=False)
def collect_recent_events(team_id, days=18):
    rows = []
    seen = set()
    today = date.today()
    for offset in range(1, days + 1):
        day = (today - timedelta(days=offset)).isoformat()
        try:
            events = football.events_day(day)
        except Exception:
            continue
        for event in events:
            eid = str(event.get("idEvent", ""))
            if not eid or eid in seen:
                continue
            row = event_to_row(event, team_id)
            if row:
                seen.add(eid)
                rows.append(row)
        if len(rows) >= 10:
            break
        time.sleep(0.03)
    rows.sort(key=lambda x: x.get("date") or "")
    return rows[-12:]

@st.cache_data(ttl=900, show_spinner=False)
def collect_h2h_from_web(home_name, away_name):
    return {
        "provider": "Public Internet search",
        "results": public_web_search(
            f"{home_name} {away_name} head to head previous meetings results",
            max_results=8,
        ),
    }

# ============================================================
# FREE INTERNET SEARCH — NO SEARCH API KEY
# ============================================================

class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = False
    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self.skip = True
    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"}:
            self.skip = False
    def handle_data(self, data):
        if not self.skip:
            text = " ".join(data.split())
            if text:
                self.parts.append(text)
    def text(self):
        return " ".join(self.parts)

def parse_ddg_results(page):
    results = []
    # DDG HTML commonly uses result__a and result__snippet.
    links = re.findall(
        r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        page,
        flags=re.I | re.S,
    )
    snippets = re.findall(
        r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>',
        page,
        flags=re.I | re.S,
    )
    for i, (url, title_html) in enumerate(links):
        title = re.sub("<.*?>", " ", title_html)
        title = html.unescape(" ".join(title.split()))
        snippet = ""
        if i < len(snippets):
            snippet = re.sub("<.*?>", " ", snippets[i])
            snippet = html.unescape(" ".join(snippet.split()))
        if url.startswith("//duckduckgo.com/l/?"):
            m = re.search(r"uddg=([^&]+)", url)
            if m:
                from urllib.parse import unquote
                url = unquote(m.group(1))
        results.append({"title": title, "url": url, "content": snippet})
    return results

def ddg_search(query, max_results=5):
    url = "https://html.duckduckgo.com/html/"
    response = requests.get(
        url,
        params={"q": query},
        timeout=12,
        headers={"User-Agent": "Mozilla/5.0 MATCHLAB research"},
    )
    response.raise_for_status()
    return parse_ddg_results(response.text)[:max_results]

def google_news_search(query, max_results=5):
    url = "https://news.google.com/rss/search"
    response = requests.get(
        url,
        params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
        timeout=12,
        headers={"User-Agent": "Mozilla/5.0 MATCHLAB research"},
    )
    response.raise_for_status()
    import xml.etree.ElementTree as ET
    root = ET.fromstring(response.text)
    out = []
    for item in root.findall(".//item")[:max_results]:
        title = item.findtext("title") or ""
        link = item.findtext("link") or ""
        desc = item.findtext("description") or ""
        desc = re.sub("<.*?>", " ", desc)
        out.append({
            "title": html.unescape(" ".join(title.split())),
            "url": link,
            "content": html.unescape(" ".join(desc.split())),
        })
    return out

@st.cache_data(ttl=600, show_spinner=False)
def public_web_search(query, max_results=5):
    # Query both public sources before truncating. The previous implementation
    # returned as soon as the first source filled the quota, so Google News was
    # effectively ignored whenever DuckDuckGo returned enough rows.
    results = []
    seen = set()
    per_source = max(2, max_results)
    for searcher in (ddg_search, google_news_search):
        try:
            found = searcher(query, max_results=per_source)
        except Exception:
            found = []
        for item in found:
            key = item.get("url") or item.get("title")
            if key and key not in seen:
                seen.add(key)
                results.append(item)
    return results[:max_results]

@st.cache_data(ttl=600, show_spinner=False)
def public_web_research(home_name, away_name, match_date, competition_hint=""):
    queries = [
        f'"{home_name}" "{away_name}" {match_date} football',
        f'"{home_name}" latest news injuries lineup',
        f'"{away_name}" latest news injuries lineup',
        f'"{home_name}" recent results form goals',
        f'"{away_name}" recent results form goals',
        f'"{home_name}" "{away_name}" head to head',
    ]
    all_results = []
    seen = set()
    for q in queries:
        try:
            found = public_web_search(q, max_results=4)
        except Exception:
            found = []
        for r in found:
            key = r.get("url") or r.get("title")
            if key in seen:
                continue
            seen.add(key)
            r["query"] = q
            all_results.append(r)
            if len(all_results) >= 20:
                break
        if len(all_results) >= 20:
            break

    return {
        "provider": "Public Internet search (DuckDuckGo + Google News RSS)",
        "results": all_results,
        "queries": queries,
    }

def research_text(web_research, max_chars=9000):
    chunks = []
    for r in web_research.get("results", []):
        chunks.append(
            f"{r.get('title','')} | {r.get('content','')} | {r.get('url','')}"
        )
    return "\n".join(chunks)[:max_chars]

def web_injury_items(web_research, team_name):
    """Conservative extraction: only label as a reported issue when search text contains clear keywords."""
    keywords = ("injury", "injured", "out", "suspended", "suspension", "doubt", "unavailable", "miss")
    team = norm(team_name)
    items = []
    for r in web_research.get("results", []):
        text = norm(f"{r.get('title','')} {r.get('content','')}")
        if team in text and any(k in text for k in keywords):
            items.append({
                "player": "Reported availability issue",
                "type": "Web report",
                "reason": r.get("title") or r.get("content", "")[:180],
                "source": r.get("url", ""),
            })
    return items[:6]

# ============================================================
# WEB FORM PARSING
# ============================================================

def profile_from_web_if_needed(profile, web_research, team_name):
    if profile["matches"] >= 3:
        return profile

    # Do not fabricate scores from prose. Instead add a small confidence penalty
    # later and use conservative league-neutral priors.
    return profile


# ============================================================
# ADVANCED TEAM NEWS + FIXTURE CONGESTION ENGINE
# ============================================================

ROTATION_TERMS = (
    "rotate", "rotation", "rest players", "rested", "resting", "rotated",
    "changes expected", "weakened side", "squad rotation", "manage minutes",
    "minutes restriction", "rest key", "key players rested",
)
BIG_MATCH_TERMS = (
    "champions league", "europa league", "conference league", "fa cup",
    "carabao cup", "copa del rey", "coppa italia", "dfb pokal", "cup final",
    "semi-final", "semifinal", "quarter-final", "quarterfinal", "final",
    "derby", "playoff", "play-off", "promotion final", "relegation six-pointer",
)
ABSENCE_TERMS = (
    "injury", "injured", "suspended", "suspension", "unavailable", "doubtful",
    "misses", "miss", "ruled out", "out of", "fitness test", "illness",
)
LINEUP_TERMS = (
    "predicted lineup", "predicted line-up", "starting xi", "starting eleven",
    "confirmed lineup", "confirmed line-up", "team news", "likely lineup",
)


def _source_quality_score(result):
    url = norm(result.get("url", ""))
    title = norm(result.get("title", ""))
    score = 0
    if any(x in url for x in (".gov", "uefa.com", "fifa.com", "premierleague.com")):
        score += 5
    if any(x in url for x in ("bbc.", "skysports.com", "espn.com", "theguardian.com", "reuters.com", "apnews.com")):
        score += 4
    if any(x in title for x in ("official", "confirmed", "press conference")):
        score += 2
    return score


@st.cache_data(ttl=600, show_spinner=False)
def research_team_news(home_team, away_team, match_date, competition_hint=""):
    """Targeted team-news research. Facts stay evidence-backed; weak snippets never become fake player data."""
    queries = [
        f'"{home_team}" team news injuries suspensions {match_date}',
        f'"{away_team}" team news injuries suspensions {match_date}',
        f'"{home_team}" predicted lineup starting XI {match_date}',
        f'"{away_team}" predicted lineup starting XI {match_date}',
        f'"{home_team}" rotation rest players next match',
        f'"{away_team}" rotation rest players next match',
        f'"{home_team}" manager press conference team news',
        f'"{away_team}" manager press conference team news',
    ]
    results=[]; seen=set()
    for q in queries:
        try:
            found=public_web_search(q,max_results=6)
        except Exception:
            found=[]
        for r in found:
            key=r.get("url") or r.get("title")
            if not key or key in seen:
                continue
            seen.add(key)
            item=dict(r); item["query"]=q; item["quality"]=_source_quality_score(item)
            results.append(item)
    results.sort(key=lambda x:(x.get("quality",0), len(x.get("content", ""))), reverse=True)

    def team_pack(team):
        tn=norm(team); facts=[]; rotation=[]; lineup=[]; absences=[]
        for r in results:
            text=norm(f'{r.get("title","")} {r.get("content","")}')
            if tn not in text:
                continue
            if any(k in text for k in ABSENCE_TERMS):
                absences.append(r)
            if any(k in text for k in ROTATION_TERMS):
                rotation.append(r)
            if any(k in text for k in LINEUP_TERMS):
                lineup.append(r)
            if any(k in text for k in ABSENCE_TERMS+ROTATION_TERMS+LINEUP_TERMS):
                facts.append({"Topic":r.get("query",""),"Source":r.get("title",""),"Evidence":r.get("content","")[:450],"URL":r.get("url","")})
        return {
            "facts":facts[:10],
            "absence_sources":absences[:6],
            "rotation_sources":rotation[:6],
            "lineup_sources":lineup[:6],
            "evidence_count":len(facts),
        }
    return {"home":team_pack(home_team),"away":team_pack(away_team),"results":results[:30],"queries":queries}


def _event_date(event):
    raw=str(event.get("dateEvent") or event.get("date") or "")[:10]
    try:
        return date.fromisoformat(raw)
    except Exception:
        return None


def _fixture_label(event):
    return f"{event.get('strHomeTeam','?')} vs {event.get('strAwayTeam','?')}"


def _competition_importance(name):
    text=norm(name)
    if any(k in text for k in ("champions league", "europa league", "conference league")):
        return 5
    if any(k in text for k in ("final", "semi-final", "semifinal", "quarter-final", "quarterfinal", "playoff", "play-off", "derby")):
        return 5
    if any(k in text for k in ("cup", "copa", "coppa", "pokal")):
        return 3
    return 1


@st.cache_data(ttl=900, show_spinner=False)
def future_fixture_research(team_name, team_id, after_date, competition_hint="", days=21):
    """Find the team's next fixtures and flag congestion / likely rotation spots."""
    events=[]
    if team_id:
        try:
            events.extend(football.eventsnext(team_id) or [])
        except Exception:
            pass
    clean=[]; seen=set()
    for e in events:
        eid=str(e.get("idEvent") or _fixture_label(e))
        if eid in seen: continue
        seen.add(eid)
        d=_event_date(e)
        if not d or d <= after_date or d > after_date+timedelta(days=days):
            continue
        clean.append(e)
    clean.sort(key=lambda e:_event_date(e) or date.max)

    # Web fallback catches competitions missing from the free schedule endpoint.
    if len(clean)<3:
        queries=[
            f'"{team_name}" fixtures next matches {after_date.isoformat()}',
            f'"{team_name}" upcoming fixtures next 3 weeks',
        ]
        for q in queries:
            try: found=public_web_search(q,max_results=8)
            except Exception: found=[]
            for r in found:
                text=f"{r.get('title','')} {r.get('content','')}"
                dates=_extract_iso_dates(text)
                for d in dates:
                    if after_date < d <= after_date+timedelta(days=days):
                        clean.append({"strHomeTeam":team_name,"strAwayTeam":"Upcoming opponent (web)","dateEvent":d.isoformat(),"strLeague":"Web schedule","strEvent":"Web-researched fixture","_web_source":r})
                        break
            if len(clean)>=3: break

    dedup=[]; seen_dates=set()
    for e in sorted(clean,key=lambda e:_event_date(e) or date.max):
        d=_event_date(e)
        key=(d, norm(e.get("strHomeTeam")), norm(e.get("strAwayTeam")))
        if key in seen_dates: continue
        seen_dates.add(key); dedup.append(e)

    rows=[]
    for e in dedup[:8]:
        d=_event_date(e)
        gap=(d-after_date).days if d else None
        comp=e.get("strLeague") or "Unknown competition"
        rows.append({
            "date":d.isoformat() if d else "",
            "fixture":_fixture_label(e),
            "competition":comp,
            "days_after_target":gap,
            "importance":_competition_importance(comp),
            "source":("web" if e.get("_web_source") else "TheSportsDB"),
            "url":(e.get("_web_source") or {}).get("url","") if e.get("_web_source") else "",
        })

    next_big=next((r for r in rows if r["importance"]>=3),None)
    return {"team":team_name,"fixtures":rows,"next_big":next_big}


def schedule_adjustment(schedule, news_pack):
    """Small, capped rotation/fatigue effect. Schedule context cannot overpower core team strength."""
    fixtures=schedule.get("fixtures",[])
    if not fixtures:
        return 1.0, "No reliable upcoming-fixture congestion signal."
    near=[f for f in fixtures if f.get("days_after_target") is not None and f["days_after_target"]<=4]
    big=[f for f in fixtures if f.get("importance",1)>=3 and f.get("days_after_target",99)<=7]
    text=" ".join(norm(f"{x.get('title','')} {x.get('content','')}" ) for x in news_pack.get("rotation_sources",[]))
    explicit=any(k in text for k in ROTATION_TERMS)
    if explicit and big:
        return 0.972, f"Explicit rotation/rest reporting plus a major fixture within 7 days ({big[0]['fixture']})."
    if near and big:
        return 0.982, f"Compressed schedule with a major upcoming fixture ({big[0]['fixture']})."
    if near:
        return 0.990, "Short turnaround detected; a small fatigue adjustment is applied."
    return 1.0, "No meaningful fixture-congestion penalty applied."


def market_value_rows(matrix_result, odds_map):
    """Value analysis for 1X2 + totals + BTTS. Each market removes its own overround."""
    specs={
        "Home Win":("1X2",matrix_result.get("home_win")),
        "Draw":("1X2",matrix_result.get("draw")),
        "Away Win":("1X2",matrix_result.get("away_win")),
        "Over 1.5":("O/U 1.5",matrix_result.get("over_15")),
        "Under 1.5":("O/U 1.5",matrix_result.get("under_15")),
        "Over 2.5":("O/U 2.5",matrix_result.get("over_25")),
        "Under 2.5":("O/U 2.5",matrix_result.get("under_25")),
        "Over 3.5":("O/U 3.5",matrix_result.get("over_35")),
        "Under 3.5":("O/U 3.5",matrix_result.get("under_35")),
        "BTTS Yes":("BTTS",matrix_result.get("btts_yes")),
        "BTTS No":("BTTS",matrix_result.get("btts_no")),
    }
    valid={k:safe_float(v) for k,v in odds_map.items() if safe_float(v) and safe_float(v)>1.0}
    by_market={}
    for label,(market,p) in specs.items():
        if p is None or label not in valid: continue
        by_market.setdefault(market,[]).append((label,p,valid[label]))
    rows=[]
    required_counts={"1X2":2,"O/U 1.5":2,"O/U 2.5":2,"O/U 3.5":2,"BTTS":2}
    for market,items in by_market.items():
        # Never call a single supplied side a complete market; doing so would
        # falsely make its implied probability 100%.
        if len(items) < required_counts.get(market,2):
            continue
        raw=[1/o for _,_,o in items]; total=sum(raw)
        if total<=0: continue
        for (label,p,o),imp in zip(items,raw):
            market_p=imp/total
            edge=p-market_p
            ev=p*o-1
            rows.append({
                "Market":market,"Outcome":label,"Model Probability":p,
                "Market Probability":market_p,"Edge":edge,"Odds":o,
                "Fair Odds":fair_odds(p),"Expected Return":ev,
                "Potential Value":edge>=0.05 and ev>=0.05,
            })
    return sorted(rows,key=lambda r:(r["Potential Value"],r["Edge"],r["Model Probability"]),reverse=True)


def best_bet_recommendation(value_rows, coverage, uncertainty, provisional=False):
    if not value_rows:
        return {"decision":"NO BET","reason":"No bookmaker odds supplied for a supported market.","row":None}
    eligible=[]
    for r in value_rows:
        threshold=0.55 if r["Market"]!="1X2" else 0.48
        if (r["Potential Value"] and r["Model Probability"]>=threshold and
            coverage>=70 and uncertainty<=0.18 and not provisional):
            eligible.append(r)
    if not eligible:
        best=max(value_rows,key=lambda r:r["Edge"])
        return {"decision":"NO BET","reason":("No market cleared MATCHLAB's probability, value, coverage and uncertainty gates."),"row":best}
    pick=max(eligible,key=lambda r:(r["Expected Return"],r["Edge"],r["Model Probability"]))
    return {"decision":"BEST VALUE OPTION","reason":"Highest risk-adjusted expected return among markets that cleared the safety gates.","row":pick}


def model_uncertainty(coverage, profiles, matrix_result, provisional=False, news_count=0):
    gap=sorted([matrix_result["home_win"],matrix_result["draw"],matrix_result["away_win"]],reverse=True)
    spread=gap[0]-gap[1]
    u=0.24
    u -= max(0,coverage-50)*0.0015
    u -= min(profiles[0].get("matches",0),profiles[1].get("matches",0))*0.004
    u -= min(news_count,10)*0.002
    u += max(0,0.08-spread)*0.25
    if provisional: u += 0.08
    return clamp(u,0.05,0.35)

# ============================================================
# ODDS INPUT — USER SUPPLIED, NEVER A MODEL INPUT
# ============================================================

def user_odds_value_analysis(matrix_result, home_team, away_team, home_odds, draw_odds, away_odds):
    return market_value_rows(matrix_result, {
        "Home Win":home_odds, "Draw":draw_odds, "Away Win":away_odds
    })


# ============================================================
# LIVE STADIUM WEATHER + MOTIVATION RESEARCH
# ============================================================

WEATHER_CODE_LABELS = {
    0: "Clear", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Freezing fog", 51: "Light drizzle", 53: "Drizzle",
    55: "Heavy drizzle", 61: "Light rain", 63: "Rain", 65: "Heavy rain",
    66: "Freezing rain", 67: "Heavy freezing rain", 71: "Light snow",
    73: "Snow", 75: "Heavy snow", 77: "Snow grains", 80: "Rain showers",
    81: "Rain showers", 82: "Heavy rain showers", 85: "Snow showers",
    86: "Heavy snow showers", 95: "Thunderstorm", 96: "Thunderstorm + hail",
    99: "Severe thunderstorm + hail",
}

def _geocode_stadium(venue_name, home_team):
    """Resolve the host stadium to coordinates using the free Nominatim service."""
    candidates = []
    if venue_name and "unknown" not in norm(venue_name) and "not independently" not in norm(venue_name):
        candidates.extend([f"{venue_name}, {home_team}", venue_name])
    candidates.extend([f"{home_team} stadium", home_team])
    for query in candidates:
        try:
            response = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": query, "format": "jsonv2", "limit": 3},
                timeout=10,
                headers={"User-Agent": "MATCHLAB-Live/1.0 (football prediction research)"},
            )
            response.raise_for_status()
            rows = response.json() or []
            if rows:
                row = rows[0]
                return {
                    "latitude": float(row["lat"]),
                    "longitude": float(row["lon"]),
                    "display_name": row.get("display_name", query),
                    "query": query,
                }
        except Exception:
            continue
    return None

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_stadium_weather(venue_name, home_team, fixture_datetime, match_date):
    """Fetch current/future host-stadium weather from Open-Meteo. No API key."""
    geo = _geocode_stadium(venue_name, home_team)
    if not geo:
        return {"available": False, "reason": "Host stadium could not be geocoded."}

    target = match_date.isoformat()
    today = date.today()
    days_ahead = (match_date - today).days
    if days_ahead < 0:
        endpoint = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": geo["latitude"], "longitude": geo["longitude"],
            "start_date": target, "end_date": target, "daily": "weathercode,temperature_2m_mean,precipitation_sum,windspeed_10m_max",
            "timezone": "auto",
        }
        mode = "historical"
    else:
        endpoint = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": geo["latitude"], "longitude": geo["longitude"],
            "start_date": target, "end_date": target,
            "daily": "weathercode,temperature_2m_mean,precipitation_sum,windspeed_10m_max",
            "timezone": "auto",
            "forecast_days": 16,
        }
        mode = "forecast"
    try:
        response = requests.get(endpoint, params=params, timeout=12, headers={"User-Agent": "MATCHLAB-Live/1.0"})
        response.raise_for_status()
        data = response.json()
        daily = data.get("daily") or {}
        if not daily.get("time"):
            return {"available": False, "reason": "Weather provider returned no data for this match date.", "geo": geo}
        code = int((daily.get("weathercode") or [0])[0])
        temp = safe_float((daily.get("temperature_2m_mean") or [None])[0])
        precip = safe_float((daily.get("precipitation_sum") or [None])[0], 0.0)
        wind = safe_float((daily.get("windspeed_10m_max") or [None])[0], 0.0)
        label = WEATHER_CODE_LABELS.get(code, "Variable conditions")
        return {
            "available": True, "mode": mode, "label": label, "weather_code": code,
            "temperature_c": temp, "precipitation_mm": precip, "wind_kmh": wind,
            "geo": geo, "source": "Open-Meteo + OpenStreetMap Nominatim",
            "environment_factor": weather_environment_factor(label, precip, wind, temp),
        }
    except Exception as exc:
        return {"available": False, "reason": f"Weather service unavailable: {type(exc).__name__}", "geo": geo}

def weather_environment_factor(label, precipitation_mm, wind_kmh, temperature_c):
    """Restrained total-goals adjustment; extreme conditions reduce scoring, not certainty."""
    factor = 1.0
    if precipitation_mm >= 15: factor *= 0.94
    elif precipitation_mm >= 6: factor *= 0.975
    if wind_kmh >= 45: factor *= 0.92
    elif wind_kmh >= 30: factor *= 0.96
    if temperature_c is not None:
        if temperature_c >= 33: factor *= 0.965
        elif temperature_c <= 2: factor *= 0.975
    if "snow" in label.lower() or "thunderstorm" in label.lower(): factor *= 0.965
    return clamp(factor, 0.86, 1.02)

MOTIVATION_POSITIVE = ("must win", "need to win", "win to", "title race", "promotion", "qualif", "survival", "avoid relegation", "relegation battle", "final", "semi-final", "semifinal", "decider", "second leg")
MOTIVATION_NEGATIVE = ("nothing to play for", "dead rubber", "already eliminated", "already relegated", "rest players", "rotate", "rotation", "rotating", "rested")

def research_motivation(home_team, away_team, competition_hint, web_research):
    """Score motivation only from explicit public evidence; weak/no evidence stays neutral."""
    queries = [
        f'"{home_team}" motivation must win {competition_hint}',
        f'"{away_team}" motivation must win {competition_hint}',
        f'"{home_team}" rotation rest players {competition_hint}',
        f'"{away_team}" rotation rest players {competition_hint}',
        f'"{home_team}" "{away_team}" stakes qualification relegation title',
    ]
    results = []
    seen = set()
    for q in queries:
        try:
            found = public_web_search(q, max_results=4)
        except Exception:
            found = []
        for r in found:
            key = r.get("url") or r.get("title")
            if key and key not in seen:
                seen.add(key); r = dict(r); r["query"] = q; results.append(r)
    def team_score(team):
        score = 0; evidence = []
        tn = norm(team)
        for r in results:
            text = norm(f"{r.get('title','')} {r.get('content','')}")
            if tn not in text:
                continue
            pos = [p for p in MOTIVATION_POSITIVE if p in text]
            neg = [n for n in MOTIVATION_NEGATIVE if n in text]
            if pos or neg:
                score += min(2, len(pos)) - min(2, len(neg))
                evidence.append({"Source": r.get("title", ""), "Signal": ", ".join(pos + neg), "Evidence": r.get("content", "")[:350], "URL": r.get("url", "")})
        level = "High" if score >= 2 else "Low" if score <= -2 else "Normal/unclear"
        return {"score": clamp(score, -3, 3), "level": level, "evidence": evidence[:5]}
    return {"home": team_score(home_team), "away": team_score(away_team), "results": results[:15], "source": "Public Internet research; neutral when evidence is weak"}

def motivation_adjustments(motivation):
    hs = motivation.get("home", {}).get("score", 0)
    aw = motivation.get("away", {}).get("score", 0)
    # Small effect only: motivation should not overpower actual football strength.
    return clamp(1.0 + hs * 0.012, 0.965, 1.036), clamp(1.0 + aw * 0.012, 0.965, 1.036)


# ============================================================
# HISTORICAL / RECENT TEAM MODEL
# ============================================================

def completed_fixture(fixture):
    status = fixture.get("fixture", {}).get("status", {}).get("short")
    return status in {"FT", "AET", "PEN"}


def extract_match_row(fixture, team_id):
    if not completed_fixture(fixture):
        return None

    home = fixture.get("teams", {}).get("home", {})
    away = fixture.get("teams", {}).get("away", {})
    goals = fixture.get("goals", {})

    hg = goals.get("home")
    ag = goals.get("away")

    if hg is None or ag is None:
        return None

    if home.get("id") == team_id:
        return {
            "date": fixture.get("fixture", {}).get("date"),
            "team_id": team_id,
            "opponent_id": away.get("id"),
            "venue": "home",
            "gf": float(hg),
            "ga": float(ag),
            "result": "W" if hg > ag else "D" if hg == ag else "L",
            "competition": fixture.get("league", {}).get("name", ""),
        }

    if away.get("id") == team_id:
        return {
            "date": fixture.get("fixture", {}).get("date"),
            "team_id": team_id,
            "opponent_id": home.get("id"),
            "venue": "away",
            "gf": float(ag),
            "ga": float(hg),
            "result": "W" if ag > hg else "D" if ag == hg else "L",
            "competition": fixture.get("league", {}).get("name", ""),
        }

    return None


def weighted_average(values, weights):
    if not values:
        return None
    total_weight = sum(weights)
    if total_weight <= 0:
        return sum(values) / len(values)
    return sum(v * w for v, w in zip(values, weights)) / total_weight


def build_recent_team_profile(fixtures, team_id):
    rows = []

    for fixture in fixtures:
        row = extract_match_row(fixture, team_id)
        if row:
            rows.append(row)

    rows.sort(key=lambda x: x.get("date") or "")
    rows = rows[-15:]

    if not rows:
        return {
            "matches": 0,
            "home_matches": 0,
            "away_matches": 0,
            "gf": 1.35,
            "ga": 1.35,
            "home_gf": 1.45,
            "home_ga": 1.20,
            "away_gf": 1.25,
            "away_ga": 1.35,
            "points": 1.35,
            "recent_results": [],
        }

    weights = []
    n = len(rows)
    for index in range(n):
        weights.append(0.82 ** (n - 1 - index))

    gf = weighted_average([r["gf"] for r in rows], weights)
    ga = weighted_average([r["ga"] for r in rows], weights)
    points = weighted_average(
        [3 if r["result"] == "W" else 1 if r["result"] == "D" else 0 for r in rows],
        weights,
    )

    home_rows = [r for r in rows if r["venue"] == "home"]
    away_rows = [r for r in rows if r["venue"] == "away"]

    def venue_metric(venue_rows, key, fallback):
        if not venue_rows:
            return fallback
        venue_weights = []
        m = len(venue_rows)
        for index in range(m):
            venue_weights.append(0.85 ** (m - 1 - index))
        return weighted_average(
            [r[key] for r in venue_rows],
            venue_weights,
        )

    return {
        "matches": len(rows),
        "home_matches": len(home_rows),
        "away_matches": len(away_rows),
        "gf": gf,
        "ga": ga,
        "home_gf": venue_metric(home_rows, "gf", 1.45),
        "home_ga": venue_metric(home_rows, "ga", 1.20),
        "away_gf": venue_metric(away_rows, "gf", 1.25),
        "away_ga": venue_metric(away_rows, "ga", 1.35),
        "points": points,
        "recent_results": [r["result"] for r in rows[-5:]],
        "rows": rows,
    }


def profile_strength(profile):
    # Convert recent scoring/conceding into restrained multipliers.
    attack = clamp(
        1.0
        + (profile["gf"] - 1.35) * 0.09
        + (profile["points"] - 1.35) * 0.025,
        0.82,
        1.18,
    )

    defense = clamp(
        1.0 + (profile["ga"] - 1.35) * 0.075,
        0.84,
        1.16,
    )

    return attack, defense

# ============================================================
# CONTEXT ADJUSTMENTS
# ============================================================

FORMATION = {
    "4-3-3": (1.015, 1.000),
    "4-2-3-1": (1.008, 0.995),
    "4-4-2": (1.000, 1.000),
    "3-4-3": (1.018, 1.012),
    "3-5-2": (1.005, 0.990),
    "5-3-2": (0.985, 0.975),
    "5-4-1": (0.970, 0.965),
    "4-5-1": (0.985, 0.975),
}

MOTIVATION = {
    "Very Low": 0.975,
    "Low": 0.988,
    "Normal": 1.000,
    "High": 1.012,
    "Very High": 1.025,
}

WEATHER = {
    "Clear": 1.000,
    "Cloudy": 1.000,
    "Light Rain": 0.985,
    "Heavy Rain": 0.950,
    "Strong Wind": 0.940,
    "Extreme Heat": 0.965,
    "Snow": 0.930,
}

PITCH = {
    "Excellent": 1.000,
    "Good": 1.000,
    "Average": 0.985,
    "Poor": 0.960,
}


def form_adjustments(profile):
    attack, defense = profile_strength(profile)
    return attack, defense


def context_multiplier(
    formation,
    motivation,
    absences,
    absence_type,
    absence_importance,
    profile,
    rest_days,
    aggregate_for,
    aggregate_against,
    second_leg,
):
    attack, defense = FORMATION.get(
        formation,
        (1.0, 1.0),
    )

    attack *= MOTIVATION.get(
        motivation,
        1.0,
    )

    if profile:
        form_attack, form_defense = form_adjustments(profile)
        attack *= 1 + (form_attack - 1) * 0.70
        defense *= 1 + (form_defense - 1) * 0.70

    importance = absence_importance / 100.0
    count = min(int(absences), 4)

    role_attack_penalty = {
        "Attackers": 0.030,
        "Midfielders": 0.018,
        "Defenders": 0.006,
        "Goalkeeper": 0.004,
        "Mixed": 0.018,
        "Not specified": 0.010,
    }.get(absence_type, 0.010)

    role_defence_penalty = {
        "Attackers": 0.003,
        "Midfielders": 0.012,
        "Defenders": 0.028,
        "Goalkeeper": 0.040,
        "Mixed": 0.020,
        "Not specified": 0.010,
    }.get(absence_type, 0.010)

    attack *= 1 - role_attack_penalty * count * importance
    defense *= 1 + role_defence_penalty * count * importance

    # Fatigue: 5 days is treated as neutral.
    rest_effect = clamp(
        (rest_days - 5) * 0.006,
        -0.025,
        0.018,
    )

    attack *= 1 + rest_effect
    defense *= 1 - rest_effect * 0.45

    if second_leg:
        if aggregate_for < aggregate_against:
            attack *= 1.045
            defense *= 1.012
        elif aggregate_for > aggregate_against:
            attack *= 0.985
            defense *= 0.992
        else:
            attack *= 1.015

    return (
        clamp(attack, 0.82, 1.20),
        clamp(defense, 0.82, 1.20),
    )

# ============================================================
# EXPECTED GOALS
# ============================================================

def predict_xg(
    home_profile,
    away_profile,
    home_context,
    away_context,
    venue,
    weather,
    pitch,
    environment_factor_override=None,
):
    home_attack, home_defense = profile_strength(home_profile)
    away_attack, away_defense = profile_strength(away_profile)

    # Location-specific recent rates are more useful than a fixed team list.
    if venue == "Away team venue":
        base_home = 1.25
        base_away = 1.45
        home_attack_rate = away_profile["away_gf"] / 1.25
        away_attack_rate = home_profile["home_gf"] / 1.45
        home_defence_rate = away_profile["away_ga"] / 1.35
        away_defence_rate = home_profile["home_ga"] / 1.20

    elif venue == "Neutral venue":
        base_home = 1.35
        base_away = 1.35
        home_attack_rate = home_attack
        away_attack_rate = away_attack
        home_defence_rate = home_defense
        away_defence_rate = away_defense

    else:
        base_home = 1.50
        base_away = 1.20
        home_attack_rate = home_profile["home_gf"] / 1.45
        away_attack_rate = away_profile["away_gf"] / 1.25
        home_defence_rate = home_profile["home_ga"] / 1.20
        away_defence_rate = away_profile["away_ga"] / 1.35

    home_attack_rate = clamp(home_attack_rate, 0.55, 1.60)
    away_attack_rate = clamp(away_attack_rate, 0.55, 1.60)
    home_defence_rate = clamp(home_defence_rate, 0.55, 1.60)
    away_defence_rate = clamp(away_defence_rate, 0.55, 1.60)

    home_xg = (
        base_home
        * home_attack_rate
        * away_defence_rate
        * home_context[0]
        * away_context[1]
    )

    away_xg = (
        base_away
        * away_attack_rate
        * home_defence_rate
        * away_context[0]
        * home_context[1]
    )

    environment_factor = (
        WEATHER.get(weather, 1.0)
        * PITCH.get(pitch, 1.0)
    )
    if environment_factor_override is not None:
        environment_factor = clamp(float(environment_factor_override), 0.86, 1.02)

    home_xg *= environment_factor
    away_xg *= environment_factor

    return (
        clamp(home_xg, 0.05, 4.50),
        clamp(away_xg, 0.05, 4.50),
    )

# ============================================================
# DIXON-COLES / POISSON
# ============================================================

def poisson_pmf(k, lam):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def calculate_matrix(home_xg, away_xg, max_goals=10, rho=-0.05):
    matrix = []
    total = 0.0

    for hg in range(max_goals + 1):
        row = []
        for ag in range(max_goals + 1):
            p = (
                poisson_pmf(hg, home_xg)
                * poisson_pmf(ag, away_xg)
            )

            if hg == 0 and ag == 0:
                tau = 1 - home_xg * away_xg * rho
            elif hg == 0 and ag == 1:
                tau = 1 + home_xg * rho
            elif hg == 1 and ag == 0:
                tau = 1 + away_xg * rho
            elif hg == 1 and ag == 1:
                tau = 1 - rho
            else:
                tau = 1.0

            p *= max(tau, 0.01)
            row.append(p)
            total += p

        matrix.append(row)

    if total <= 0:
        raise RuntimeError("Probability matrix could not be normalized.")

    matrix = [
        [p / total for p in row]
        for row in matrix
    ]

    home_win = 0.0
    draw = 0.0
    away_win = 0.0

    markets = Counter()
    scorelines = []
    home_clean = 0.0
    away_clean = 0.0

    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            p = matrix[hg][ag]

            if hg > ag:
                home_win += p
            elif hg == ag:
                draw += p
            else:
                away_win += p

            goals = hg + ag

            for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
                if goals > line:
                    markets[f"over_{str(line).replace('.', '')}"] += p

            if hg > 0 and ag > 0:
                markets["btts_yes"] += p

            if ag == 0:
                home_clean += p

            if hg == 0:
                away_clean += p

            scorelines.append((hg, ag, p))

    scorelines.sort(key=lambda x: x[2], reverse=True)

    return {
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "over_05": markets["over_05"],
        "over_15": markets["over_15"],
        "over_25": markets["over_25"],
        "over_35": markets["over_35"],
        "over_45": markets["over_45"],
        "under_05": 1 - markets["over_05"],
        "under_15": 1 - markets["over_15"],
        "under_25": 1 - markets["over_25"],
        "under_35": 1 - markets["over_35"],
        "under_45": 1 - markets["over_45"],
        "btts_yes": markets["btts_yes"],
        "btts_no": 1 - markets["btts_yes"],
        "home_clean_sheet": home_clean,
        "away_clean_sheet": away_clean,
        "home_or_draw": home_win + draw,
        "draw_or_away": draw + away_win,
        "home_or_away": home_win + away_win,
        "top_scores": scorelines[:10],
        "matrix": matrix,
    }


def poisson_sample(lam, rng):
    if lam <= 0:
        return 0

    threshold = math.exp(-lam)
    product = 1.0
    k = 0

    while product > threshold:
        k += 1
        product *= rng.random()

    return k - 1


def run_monte_carlo(home_xg, away_xg, n_sims, seed=42):
    rng = random.Random(seed)
    counts = Counter()
    score_counter = Counter()
    total_goals = 0

    for _ in range(n_sims):
        hg = poisson_sample(home_xg, rng)
        ag = poisson_sample(away_xg, rng)

        score_counter[(hg, ag)] += 1
        total_goals += hg + ag

        if hg > ag:
            counts["home_win"] += 1
        elif hg == ag:
            counts["draw"] += 1
        else:
            counts["away_win"] += 1

        if hg + ag >= 1:
            counts["over_05"] += 1
        if hg + ag >= 2:
            counts["over_15"] += 1
        if hg + ag >= 3:
            counts["over_25"] += 1
        if hg + ag >= 4:
            counts["over_35"] += 1
        if hg + ag >= 5:
            counts["over_45"] += 1
        if hg > 0 and ag > 0:
            counts["btts_yes"] += 1

    result = {
        key: counts[key] / n_sims
        for key in [
            "home_win",
            "draw",
            "away_win",
            "over_05",
            "over_15",
            "over_25",
            "over_35",
            "over_45",
            "btts_yes",
        ]
    }

    result["samples"] = n_sims
    result["average_goals"] = total_goals / n_sims
    result["top_scores"] = [
        (hg, ag, count / n_sims)
        for (hg, ag), count in score_counter.most_common(10)
    ]

    return result

# ============================================================
# ODDS / VALUE ENGINE
# ============================================================

def parse_odds_response(odds_response):
    rows = []

    for bookmaker_block in odds_response:
        bookmaker = bookmaker_block.get("bookmaker", {})
        bookmaker_name = bookmaker.get("name", "Unknown")

        for bet in bookmaker.get("bets", []):
            bet_name = bet.get("name", "")

            if bet_name.lower() not in {
                "match winner",
                "1x2",
                "winner",
            }:
                continue

            for value in bet.get("values", []):
                label = value.get("value")
                odd = safe_float(value.get("odd"))
                if odd and label:
                    rows.append({
                        "Bookmaker": bookmaker_name,
                        "Outcome": label,
                        "Odds": odd,
                    })

    return rows


def normalize_market_probs(rows):
    """Remove bookmaker margin from a single 1X2 market."""
    raw = {}
    for row in rows:
        raw.setdefault(row["Outcome"], []).append(row["Odds"])

    best = {
        outcome: min(values)
        for outcome, values in raw.items()
        if values
    }

    implied = {
        outcome: implied_probability(odd)
        for outcome, odd in best.items()
    }

    total = sum(v for v in implied.values() if v)

    if total <= 0:
        return best, {}

    fair_market = {
        outcome: value / total
        for outcome, value in implied.items()
        if value
    }

    return best, fair_market


def value_analysis(matrix_result, odds_rows, home_team, away_team):
    if not odds_rows:
        return None

    best_odds, market_probs = normalize_market_probs(odds_rows)

    mapping = {
        "Home": home_team,
        "Draw": "Draw",
        "Away": away_team,
    }

    model_probs = {
        "Home": matrix_result["home_win"],
        "Draw": matrix_result["draw"],
        "Away": matrix_result["away_win"],
    }

    rows = []

    for market_label, model_probability in model_probs.items():
        market_team_name = mapping[market_label]

        market_probability = None
        market_odds = None

        # API-Football commonly uses Home/Draw/Away.
        for candidate in [
            market_label,
            market_team_name,
        ]:
            if candidate in market_probs:
                market_probability = market_probs[candidate]
            if candidate in best_odds:
                market_odds = best_odds[candidate]

        if market_probability is None:
            continue

        edge = model_probability - market_probability
        fair = fair_odds(model_probability)
        value_flag = (
            edge >= 0.05
            and market_odds is not None
            and market_odds > fair
        )

        rows.append({
            "Outcome": market_team_name,
            "Model Probability": model_probability,
            "Market Probability": market_probability,
            "Edge": edge,
            "Best Odds": market_odds,
            "Model Fair Odds": fair,
            "Potential Value": value_flag,
        })

    return rows

# ============================================================
# EXPLANATION ENGINE
# ============================================================

def build_explanation(
    home_team,
    away_team,
    home_profile,
    away_profile,
    home_xg,
    away_xg,
    matrix_result,
    injuries_home,
    injuries_away,
    h2h,
    home_context,
    away_context,
):
    reasons = []
    risks = []

    home_attack_edge = home_profile["gf"] - away_profile["ga"]
    away_attack_edge = away_profile["gf"] - home_profile["ga"]

    if home_attack_edge > 0.25:
        reasons.append(
            f"{home_team} has the stronger recent scoring-versus-opponent-defence profile: "
            f"{home_profile['gf']:.2f} goals per match versus {away_profile['ga']:.2f} conceded."
        )
    elif away_attack_edge > 0.25:
        reasons.append(
            f"{away_team} has the stronger recent attacking-versus-defence matchup: "
            f"{away_profile['gf']:.2f} scored versus {home_profile['ga']:.2f} conceded."
        )

    if home_profile["points"] - away_profile["points"] > 0.35:
        reasons.append(
            f"Recent weighted points favour {home_team} "
            f"({home_profile['points']:.2f} per match versus {away_profile['points']:.2f})."
        )
    elif away_profile["points"] - home_profile["points"] > 0.35:
        reasons.append(
            f"Recent weighted points favour {away_team} "
            f"({away_profile['points']:.2f} per match versus {home_profile['points']:.2f})."
        )

    if home_profile["home_gf"] > home_profile["away_gf"] + 0.15:
        reasons.append(
            f"{home_team}'s recent home scoring rate is stronger than its overall rate."
        )

    if away_profile["away_gf"] > away_profile["home_gf"] + 0.15:
        reasons.append(
            f"{away_team} has shown a meaningful away-scoring signal in its recent sample."
        )

    if home_xg > away_xg + 0.35:
        reasons.append(
            f"The combined model produces a sizeable expected-goals advantage for {home_team}: "
            f"{home_xg:.2f} xG versus {away_xg:.2f}."
        )
    elif away_xg > home_xg + 0.35:
        reasons.append(
            f"The combined model produces a sizeable expected-goals advantage for {away_team}: "
            f"{away_xg:.2f} xG versus {home_xg:.2f}."
        )

    if injuries_home:
        risks.append(
            f"{home_team} has {len(injuries_home)} reported availability issue(s); "
            "their exact importance depends on player role and minutes."
        )

    if injuries_away:
        risks.append(
            f"{away_team} has {len(injuries_away)} reported availability issue(s); "
            "their exact importance depends on player role and minutes."
        )

    if matrix_result["draw"] > 0.25:
        risks.append(
            "The draw probability is materially high, so the leading team outcome is not dominant."
        )

    if abs(home_xg - away_xg) < 0.25:
        risks.append(
            "Expected goals are close, meaning the match is structurally competitive and sensitive to small events."
        )

    if matrix_result["btts_yes"] > 0.60:
        reasons.append(
            "Both-teams-to-score probability is elevated, suggesting attacking contributions from both sides."
        )

    if matrix_result["over_25"] > 0.58:
        reasons.append(
            "The model's total-goals distribution leans toward three or more goals."
        )

    if not reasons:
        reasons.append(
            "The result is primarily driven by the combined recent-strength and expected-goals estimates rather than one extreme input."
        )

    if not risks:
        risks.append(
            "No single counter-factor dominates the model, but football outcomes remain uncertain even when the probability gap is meaningful."
        )

    return reasons, risks

# ============================================================
# DATA COVERAGE
# ============================================================

def coverage_score(home_profile, away_profile, fixture, injuries_home, injuries_away, h2h):
    score = 30

    matches = home_profile["matches"] + away_profile["matches"]
    score += min(25, matches)

    if fixture:
        score += 10

    if injuries_home is not None:
        score += 7

    if injuries_away is not None:
        score += 7

    if h2h:
        score += 5

    if fixture and fixture.get("fixture", {}).get("venue", {}).get("name"):
        score += 4

    score = int(clamp(score, 0, 100))

    if score >= 80:
        level = "High"
    elif score >= 60:
        level = "Moderate"
    else:
        level = "Limited"

    return score, level


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("MATCHLAB LIVE 10X")
    page = st.radio("Workspace", ["Analyze Fixture", "Prediction History"])
    st.divider()
    st.subheader("Data Connections")
    st.success("Free football data: TheSportsDB")
    st.success("Internet research: public search")
    st.caption(
        "No API-Football, Tavily or Serper key is required. "
        "Internet research is performed automatically during analysis. Future fixtures, rotation risk, team news, weather and market-value checks are also researched."
    )

if page == "Prediction History":
    st.markdown('<div class="title">MATCHLAB History</div>', unsafe_allow_html=True)
    st.caption("Previous predictions saved by this MATCHLAB instance.")
    history_df = load_history(200)
    if history_df.empty:
        st.info("No previous predictions have been saved yet.")
        st.stop()
    display_df = history_df.copy()
    display_df["Prediction"] = display_df.apply(
        lambda r: f"{r['predicted_outcome']} ({r['predicted_score']})", axis=1
    )
    display_df["Model"] = display_df[
        ["home_probability", "draw_probability", "away_probability"]
    ].max(axis=1).map(lambda x: f"{x * 100:.1f}%")
    columns = [
        "id","created_at","match_date","competition","home_team","away_team",
        "Prediction","Model","confidence_score","value_summary","result_checked"
    ]
    st.dataframe(display_df[columns], use_container_width=True, hide_index=True)
    st.subheader("Check a Previous Prediction")
    selected_id = st.selectbox("Prediction ID", display_df["id"].tolist())
    selected = display_df[display_df["id"] == selected_id].iloc[0]
    st.markdown(f"### {selected['home_team']} vs {selected['away_team']}")
    c1,c2,c3=st.columns(3)
    with c1: st.metric("Home", pct(selected["home_probability"]))
    with c2: st.metric("Draw", pct(selected["draw_probability"]))
    with c3: st.metric("Away", pct(selected["away_probability"]))
    if int(selected["result_checked"] or 0) == 1:
        st.success(
            f"Actual result recorded: {int(selected['actual_home_goals'])}-"
            f"{int(selected['actual_away_goals'])}"
        )
    else:
        st.info("Actual result has not been recorded yet.")
    with st.expander("Record actual result"):
        ah=st.number_input("Actual home goals",0,20,0,key="history_actual_home")
        aa=st.number_input("Actual away goals",0,20,0,key="history_actual_away")
        if st.button("Save Actual Result"):
            update_actual_result(selected_id,ah,aa)
            st.success("Actual result saved.")
            st.rerun()
    if st.button("Delete Selected Prediction"):
        delete_prediction(selected_id)
        st.success("Prediction deleted.")
        st.rerun()
    st.stop()

st.markdown("""
<div class="hero">
  <h1>⚽ MATCHLAB <span style="opacity:.65">LIVE</span></h1>
  <p>Independent football probability engine · live form · team news · future fixtures · rotation risk · weather · calibrated value</p>
</div>
""", unsafe_allow_html=True)

st.subheader("1. Real-World Fixture")
f1,f2=st.columns(2)
with f1:
    home_team_input=st.text_input("Home team",placeholder="e.g. Arsenal")
with f2:
    away_team_input=st.text_input("Away team",placeholder="e.g. Chelsea")
f3,f4=st.columns(2)
with f3:
    match_date=st.date_input("Match date",value=date.today()+timedelta(days=1))
with f4:
    competition_hint=st.text_input(
        "Competition / league hint (optional)",
        placeholder="e.g. Premier League / MLS",
    )
advanced_col1,advanced_col2=st.columns(2)
with advanced_col1:
    max_goals=st.slider("Maximum score matrix goals",5,12,9)
with advanced_col2:
    simulations=st.selectbox("Monte Carlo simulations",[5000,10000,25000],index=1)

analyze=st.button("ANALYZE FIXTURE",type="primary",use_container_width=True)

if analyze:
    if not home_team_input.strip() or not away_team_input.strip():
        st.error("Enter both teams.")
        st.stop()
    if norm(home_team_input)==norm(away_team_input):
        st.error("Home and away teams must be different.")
        st.stop()

    with st.status("MATCHLAB is researching the fixture...", expanded=True) as status:
        st.write("1/5 Verifying the real fixture with free football data and public web research...")
        try:
            fixture,home_team,away_team=find_fixture(
                home_team_input.strip(),away_team_input.strip(),match_date,competition_hint
            )
        except Exception as error:
            status.update(label="Fixture verification failed",state="error")
            st.error(str(error))
            st.info(
                "If the free football database does not cover this fixture, "
                "MATCHLAB can still use Internet research, but it will not invent a fixture."
            )
            st.stop()

        fixture_id=fixture["fixture"]["id"]
        st.write("2/5 Collecting recent form and venue-specific results...")
        home_fixtures=collect_recent_events(home_team["id"],days=18)
        away_fixtures=collect_recent_events(away_team["id"],days=18)
        home_profile=build_recent_team_profile(
            [{"fixture":{"date":x["date"]+"T00:00:00","status":{"short":"FT"}},"teams":{"home":{"id":x["team_id"] if x["venue"]=="home" else x["opponent_id"]},"away":{"id":x["opponent_id"] if x["venue"]=="home" else x["team_id"]}},"goals":{"home":x["gf"] if x["venue"]=="home" else x["ga"],"away":x["ga"] if x["venue"]=="home" else x["gf"]},"league":{"name":x["competition"]}} for x in home_fixtures],
            home_team["id"]
        )
        away_profile=build_recent_team_profile(
            [{"fixture":{"date":x["date"]+"T00:00:00","status":{"short":"FT"}},"teams":{"home":{"id":x["team_id"] if x["venue"]=="home" else x["opponent_id"]},"away":{"id":x["opponent_id"] if x["venue"]=="home" else x["team_id"]}},"goals":{"home":x["gf"] if x["venue"]=="home" else x["ga"],"away":x["ga"] if x["venue"]=="home" else x["gf"]},"league":{"name":x["competition"]}} for x in away_fixtures],
            away_team["id"]
        )

        st.write("3/6 Researching injuries, lineups, team news and future fixtures...")
        web_research=public_web_research(
            home_team["name"],away_team["name"],match_date.isoformat(),competition_hint
        )
        team_news=research_team_news(home_team["name"],away_team["name"],match_date.isoformat(),competition_hint)
        injuries_home=web_injury_items(web_research,home_team["name"])
        injuries_away=web_injury_items(web_research,away_team["name"])
        home_schedule=future_fixture_research(home_team["name"],home_team.get("id"),match_date,competition_hint)
        away_schedule=future_fixture_research(away_team["name"],away_team.get("id"),match_date,competition_hint)

        st.write("4/6 Checking host-stadium weather and match motivation...")
        weather_data=fetch_stadium_weather(
            fixture.get("fixture",{}).get("venue",{}).get("name"),
            home_team["name"],
            fixture.get("fixture",{}).get("date"),
            match_date,
        )
        motivation_data=research_motivation(
            home_team["name"],away_team["name"],competition_hint,web_research
        )
        st.write("5/6 Building the independent prediction...")
        st.write("6/6 Applying congestion, rotation and uncertainty safeguards...")
        status.update(label="Live research complete",state="complete")

    venue_name=fixture.get("fixture",{}).get("venue",{}).get("name") or "Unknown venue"
    competition=fixture.get("league",{}).get("name") or competition_hint or "Unknown competition"
    h2h=collect_h2h_from_web(home_team["name"],away_team["name"])

    motivation_attack_home,motivation_attack_away=motivation_adjustments(motivation_data)
    home_context=context_multiplier(
        "4-3-3",motivation_data.get("home",{}).get("level","Normal"),len(injuries_home),"Mixed",50,
        home_profile,5,0,0,False
    )
    away_context=context_multiplier(
        "4-3-3",motivation_data.get("away",{}).get("level","Normal"),len(injuries_away),"Mixed",50,
        away_profile,5,0,0,False
    )
    home_rotation_factor,home_rotation_note=schedule_adjustment(home_schedule,team_news.get("home",{}))
    away_rotation_factor,away_rotation_note=schedule_adjustment(away_schedule,team_news.get("away",{}))
    home_context=(home_context[0]*motivation_attack_home*home_rotation_factor,home_context[1])
    away_context=(away_context[0]*motivation_attack_away*away_rotation_factor,away_context[1])
    environment_factor=weather_data.get("environment_factor") if weather_data.get("available") else 1.0
    home_xg,away_xg=predict_xg(
        home_profile,away_profile,home_context,away_context,
        "Home team venue","Clear","Good",environment_factor_override=environment_factor
    )
    matrix=calculate_matrix(home_xg,away_xg,max_goals=max_goals,rho=-0.05)
    simulation=run_monte_carlo(home_xg,away_xg,simulations,seed=42)


    st.session_state['matchlab_analysis'] = {
        'fixture': fixture,
        'home_team': home_team,
        'away_team': away_team,
        'fixture_id': fixture_id,
        'home_profile': home_profile,
        'away_profile': away_profile,
        'web_research': web_research,
        'injuries_home': injuries_home,
        'injuries_away': injuries_away,
        'venue_name': fixture.get('fixture', {}).get('venue', {}).get('name') or 'Unknown venue',
        'competition': fixture.get('league', {}).get('name') or competition_hint or 'Unknown competition',
        'h2h': h2h,
        'home_context': home_context,
        'away_context': away_context,
        'home_xg': home_xg,
        'away_xg': away_xg,
        'matrix': matrix,
        'simulation': simulation,
        'weather_data': weather_data,
        'motivation_data': motivation_data,
        'team_news': team_news,
        'home_schedule': home_schedule,
        'away_schedule': away_schedule,
        'home_rotation_note': home_rotation_note,
        'away_rotation_note': away_rotation_note,
        'input_signature': (
            norm(home_team_input), norm(away_team_input),
            match_date.isoformat(), competition_hint.strip(),
            int(max_goals), int(simulations),
        ),
    }

analysis = st.session_state.get('matchlab_analysis')
current_signature = (
    norm(home_team_input), norm(away_team_input),
    match_date.isoformat(), competition_hint.strip(),
    int(max_goals), int(simulations),
)

if analysis and analysis.get('input_signature') == current_signature:
    fixture = analysis['fixture']
    home_team = analysis['home_team']
    away_team = analysis['away_team']
    fixture_id = analysis['fixture_id']
    home_profile = analysis['home_profile']
    away_profile = analysis['away_profile']
    web_research = analysis['web_research']
    injuries_home = analysis['injuries_home']
    injuries_away = analysis['injuries_away']
    venue_name = analysis['venue_name']
    competition = analysis['competition']
    h2h = analysis['h2h']
    home_context = analysis['home_context']
    away_context = analysis['away_context']
    home_xg = analysis['home_xg']
    away_xg = analysis['away_xg']
    matrix = analysis['matrix']
    simulation = analysis['simulation']
    weather_data = analysis.get('weather_data', {'available': False, 'reason': 'Not stored'})
    motivation_data = analysis.get('motivation_data', {'home': {'score': 0, 'level': 'Normal/unclear', 'evidence': []}, 'away': {'score': 0, 'level': 'Normal/unclear', 'evidence': []}})
    team_news = analysis.get('team_news', {'home':{},'away':{},'results':[]})
    home_schedule = analysis.get('home_schedule', {'fixtures':[]})
    away_schedule = analysis.get('away_schedule', {'fixtures':[]})
    home_rotation_note = analysis.get('home_rotation_note','')
    away_rotation_note = analysis.get('away_rotation_note','')
if analysis and analysis.get('input_signature') == current_signature:
    verified_date = fixture.get("raw_event", {}).get("verified_date")
    requested_date = fixture.get("raw_event", {}).get("requested_date")
    if verified_date and requested_date and verified_date != requested_date:
        st.warning(
            f"Web research found this fixture on {verified_date} rather than "
            f"the entered date {requested_date}. MATCHLAB will use the verified fixture date."
        )
    if fixture.get("raw_event", {}).get("provisional"):
        st.warning(
            f"Fixture accepted provisionally: {home_team['name']} vs {away_team['name']}. "
            "Free fixture verification was unavailable, so MATCHLAB did not invent a different fixture."
        )
    else:
        st.success(f"Fixture verified: {home_team['name']} vs {away_team['name']}")
    meta1,meta2,meta3=st.columns(3)
    with meta1: st.metric("Competition",competition)
    with meta2: st.metric("Venue",venue_name)
    with meta3: st.metric("Data source","Free + web")

    st.subheader("2. Live Match Conditions")
    lc1,lc2,lc3=st.columns(3)
    with lc1:
        if weather_data.get("available"):
            st.metric("Stadium weather",weather_data.get("label","Variable"))
            st.caption(f"{weather_data.get('temperature_c'):.1f}°C · {weather_data.get('precipitation_mm',0):.1f} mm rain · {weather_data.get('wind_kmh',0):.0f} km/h wind")
        else:
            st.metric("Stadium weather","Unavailable")
            st.caption(weather_data.get("reason","No reliable weather data"))
    with lc2:
        st.metric(f"{home_team['name']} motivation",motivation_data["home"]["level"])
        st.caption("Evidence-based signal; weak evidence stays neutral")
    with lc3:
        st.metric(f"{away_team['name']} motivation",motivation_data["away"]["level"])
        st.caption("Evidence-based signal; weak evidence stays neutral")

    st.subheader("3. MATCHLAB Prediction")
    probs={
        home_team["name"]:matrix["home_win"],
        "Draw":matrix["draw"],
        away_team["name"]:matrix["away_win"],
    }
    predicted_outcome=max(probs,key=probs.get)
    top_score=matrix["top_scores"][0]

    p1,p2,p3=st.columns(3)
    with p1: st.metric(home_team["name"],pct(matrix["home_win"]))
    with p2: st.metric("Draw",pct(matrix["draw"]))
    with p3: st.metric(away_team["name"],pct(matrix["away_win"]))
    x1,x2,x3=st.columns(3)
    with x1: st.metric(f"{home_team['name']} xG",f"{home_xg:.2f}")
    with x2: st.metric("Total xG",f"{home_xg+away_xg:.2f}")
    with x3: st.metric(f"{away_team['name']} xG",f"{away_xg:.2f}")
    st.markdown(f"### Most likely outcome: **{predicted_outcome}**")
    st.markdown(f"Most likely exact score: **{top_score[0]}-{top_score[1]}** ({top_score[2]*100:.2f}%)")

    reasons,risks=build_explanation(
        home_team["name"],away_team["name"],home_profile,away_profile,
        home_xg,away_xg,matrix,injuries_home,injuries_away,[],home_context,away_context
    )
    if weather_data.get("available") and weather_data.get("environment_factor",1.0) < 0.985:
        reasons.append(
            f"Host-stadium weather is restrictive ({weather_data.get('label')}, "
            f"{weather_data.get('wind_kmh',0):.0f} km/h wind, {weather_data.get('precipitation_mm',0):.1f} mm precipitation), "
            "so MATCHLAB reduces the total scoring environment rather than pretending conditions are neutral."
        )
    for team_key,team_name in (("home",home_team["name"]),("away",away_team["name"])):
        m=motivation_data.get(team_key,{})
        if m.get("level") in {"High","Low"}:
            reasons.append(f"Public reporting gives {team_name} a {m.get('level').lower()} motivation signal; the effect is deliberately capped so motivation cannot overpower performance data.")
    st.subheader("4. Why MATCHLAB Produced This Result")
    for i,r in enumerate(reasons,1):
        st.markdown(f"<div class='reason'><b>{i}.</b> {r}</div>",unsafe_allow_html=True)
    st.subheader("5. Why Other Outcomes Could Still Happen")
    for i,r in enumerate(risks,1):
        st.markdown(f"<div class='reason'><b>{i}.</b> {r}</div>",unsafe_allow_html=True)

    st.subheader("6. Recent Evidence")
    evidence_df=pd.DataFrame([
        {"Metric":"Recent matches",home_team["name"]:home_profile["matches"],away_team["name"]:away_profile["matches"]},
        {"Metric":"Weighted goals scored",home_team["name"]:round(home_profile["gf"],2),away_team["name"]:round(away_profile["gf"],2)},
        {"Metric":"Weighted goals conceded",home_team["name"]:round(home_profile["ga"],2),away_team["name"]:round(away_profile["ga"],2)},
        {"Metric":"Weighted points per match",home_team["name"]:round(home_profile["points"],2),away_team["name"]:round(away_profile["points"],2)},
        {"Metric":"Home/Away scoring rate",home_team["name"]:round(home_profile["home_gf"],2),away_team["name"]:round(away_profile["away_gf"],2)},
        {"Metric":"Home/Away conceded rate",home_team["name"]:round(home_profile["home_ga"],2),away_team["name"]:round(away_profile["away_ga"],2)},
    ])
    st.dataframe(evidence_df,use_container_width=True,hide_index=True)

    st.subheader("7. Weather & Motivation Evidence")
    if weather_data.get("available"):
        weather_df=pd.DataFrame([{
            "Stadium":venue_name,"Condition":weather_data.get("label"),
            "Temperature °C":weather_data.get("temperature_c"),
            "Rain mm":weather_data.get("precipitation_mm"),
            "Max wind km/h":weather_data.get("wind_kmh"),
            "Scoring environment":f"{weather_data.get('environment_factor',1.0):.3f}",
            "Source":weather_data.get("source",""),
        }])
        st.dataframe(weather_df,use_container_width=True,hide_index=True)
    else:
        st.info("No reliable stadium weather was available, so MATCHLAB did not apply a weather adjustment.")
    mot_rows=[]
    for key,team_name in (("home",home_team["name"]),("away",away_team["name"])):
        m=motivation_data.get(key,{})
        mot_rows.append({"Team":team_name,"Motivation":m.get("level","Normal/unclear"),"Signal score":m.get("score",0),"Evidence items":len(m.get("evidence",[]))})
    st.dataframe(pd.DataFrame(mot_rows),use_container_width=True,hide_index=True)
    mot_evidence=[]
    for key in ("home","away"):
        mot_evidence.extend(motivation_data.get(key,{}).get("evidence",[]))
    if mot_evidence:
        st.dataframe(pd.DataFrame(mot_evidence),use_container_width=True,hide_index=True)
    else:
        st.caption("No strong motivation evidence found; the model remains neutral rather than inventing a narrative.")

    st.subheader("8. Team News & Rotation Risk")
    tn1,tn2=st.columns(2)
    for col,key,team_name in ((tn1,"home",home_team["name"]),(tn2,"away",away_team["name"])):
        with col:
            pack=team_news.get(key,{})
            st.markdown(f"**{team_name}**")
            st.caption(f"Evidence items: {pack.get('evidence_count',0)} · availability reports: {len(pack.get('absence_sources',[]))} · rotation signals: {len(pack.get('rotation_sources',[]))}")
            for fact in pack.get("facts",[])[:5]:
                st.write(f"• {fact.get('Evidence','')[:300]}")
    st.caption("Team-news text is evidence only. MATCHLAB never invents a named injury or confirmed lineup from a search snippet.")

    st.subheader("9. Future Fixtures & Congestion")
    schedule_rows=[]
    for key,team_name,sched,note in (("home",home_team["name"],home_schedule,home_rotation_note),("away",away_team["name"],away_schedule,away_rotation_note)):
        for f in sched.get("fixtures",[])[:6]:
            schedule_rows.append({"Team":team_name,"Date":f.get("date"),"Fixture":f.get("fixture"),"Competition":f.get("competition"),"Days after match":f.get("days_after_target"),"Importance":f.get("importance"),"Source":f.get("source")})
        if note:
            st.info(f"{team_name}: {note}")
    if schedule_rows:
        st.dataframe(pd.DataFrame(schedule_rows),use_container_width=True,hide_index=True)
    else:
        st.warning("No reliable upcoming-fixture schedule was found; no rotation assumption was added.")

    st.subheader("10. Internet Research Used")
    st.caption(
        "MATCHLAB searched the public Internet automatically. Search evidence is supporting "
        "information; it does not directly overwrite the model probabilities."
    )
    research_rows=[]
    for r in web_research.get("results",[])[:15]:
        research_rows.append({
            "Topic": r.get("query",""),
            "Source": r.get("title",""),
            "Evidence": r.get("content","")[:500],
            "URL": r.get("url",""),
        })
    if research_rows:
        st.dataframe(pd.DataFrame(research_rows),use_container_width=True,hide_index=True)
    else:
        st.warning("Public search returned no usable results. Confidence will be reduced.")

    st.subheader("11. Availability / Team News")
    availability_rows=injuries_home+injuries_away
    if availability_rows:
        st.dataframe(pd.DataFrame(availability_rows),use_container_width=True,hide_index=True)
    else:
        st.info("No clearly reported availability issue was extracted from the public search.")

    st.subheader("12. Previous Head-to-Head Research")
    h2h_rows=[]
    for r in h2h.get("results",[])[:8]:
        h2h_rows.append({
            "Source":r.get("title",""),
            "Evidence":r.get("content","")[:600],
            "URL":r.get("url",""),
        })
    if h2h_rows:
        st.dataframe(pd.DataFrame(h2h_rows),use_container_width=True,hide_index=True)
    else:
        st.info("No usable H2H search results were found.")

    st.subheader("13. Goal Markets")
    market_df=pd.DataFrame([
        ["Over 1.5",pct(matrix["over_15"])],["Under 1.5",pct(matrix["under_15"])],
        ["Over 2.5",pct(matrix["over_25"])],["Under 2.5",pct(matrix["under_25"])],
        ["Over 3.5",pct(matrix["over_35"])],["Under 3.5",pct(matrix["under_35"])],
        ["BTTS Yes",pct(matrix["btts_yes"])],["BTTS No",pct(matrix["btts_no"])],
        ["Home Clean Sheet",pct(matrix["home_clean_sheet"])],
        ["Away Clean Sheet",pct(matrix["away_clean_sheet"])],
    ],columns=["Market","Probability"])
    st.dataframe(market_df,use_container_width=True,hide_index=True)

    st.subheader("14. Correct Score Distribution")
    score_df=pd.DataFrame([
        {"Score":f"{hg}-{ag}","Probability":pct(probability)}
        for hg,ag,probability in matrix["top_scores"][:10]
    ])
    st.dataframe(score_df,use_container_width=True,hide_index=True)

    st.subheader("15. Market Value & Best Stake Option")
    st.caption("Odds are read only after MATCHLAB has completed its prediction. They never change the model probabilities.")
    st.markdown("**1X2**")
    o1,o2,o3=st.columns(3)
    with o1: home_odds=st.number_input(f"{home_team['name']} odds",min_value=0.0,value=0.0,step=0.01,key="odds_home")
    with o2: draw_odds=st.number_input("Draw odds",min_value=0.0,value=0.0,step=0.01,key="odds_draw")
    with o3: away_odds=st.number_input(f"{away_team['name']} odds",min_value=0.0,value=0.0,step=0.01,key="odds_away")
    st.markdown("**Over / Under + BTTS**")
    cols=st.columns(6)
    labels=["Over 1.5","Under 1.5","Over 2.5","Under 2.5","Over 3.5","Under 3.5"]
    odds_map={"Home Win":home_odds,"Draw":draw_odds,"Away Win":away_odds}
    for c,label in zip(cols,labels):
        with c: odds_map[label]=st.number_input(label,min_value=0.0,value=0.0,step=0.01,key="odds_"+label.replace(" ","_").replace(".","_"))
    b1,b2=st.columns(2)
    with b1: odds_map["BTTS Yes"]=st.number_input("BTTS Yes",min_value=0.0,value=0.0,step=0.01,key="odds_btts_yes")
    with b2: odds_map["BTTS No"]=st.number_input("BTTS No",min_value=0.0,value=0.0,step=0.01,key="odds_btts_no")

    value_rows=market_value_rows(matrix,odds_map)
    if value_rows:
        value_df=pd.DataFrame(value_rows)
        shown=value_df.copy()
        for col in ("Model Probability","Market Probability","Edge"):
            shown[col]=shown[col].map(pct)
        shown["Fair Odds"]=shown["Fair Odds"].map(lambda x:f"{x:.2f}")
        shown["Odds"]=shown["Odds"].map(lambda x:f"{x:.2f}")
        shown["Expected Return"]=shown["Expected Return"].map(lambda x:f"{x*100:+.1f}%")
        st.dataframe(shown,use_container_width=True,hide_index=True)
    else:
        st.info("Enter bookmaker odds for any supported market to calculate fair probability, edge and value.")


    score,level=coverage_score(
        home_profile,away_profile,fixture,injuries_home,injuries_away,
        h2h.get("results",[])
    )
    if len(web_research.get("results",[])) < 5:
        score=int(clamp(score-10,0,100))
    if home_profile["matches"] < 3 or away_profile["matches"] < 3:
        score=int(clamp(score-10,0,100))

    # Conservative live-model gate: a high raw probability is not treated as a betting signal by itself.
    top_prob=max(probs.values())
    sorted_probs=sorted(probs.values(),reverse=True)
    probability_gap=sorted_probs[0]-sorted_probs[1]
    if fixture.get("raw_event",{}).get("provisional"):
        score=int(clamp(score-15,0,100))
    if not weather_data.get("available"):
        score=int(clamp(score-5,0,100))
    if not motivation_data.get("home",{}).get("evidence") and not motivation_data.get("away",{}).get("evidence"):
        score=int(clamp(score-5,0,100))
    if probability_gap < 0.06:
        score=int(clamp(score-8,0,100))
    # Team-news and future-fixture evidence improve context coverage without overpowering the statistical model.
    if team_news.get("home",{}).get("evidence_count",0) + team_news.get("away",{}).get("evidence_count",0) >= 4:
        score=int(clamp(score+4,0,100))
    if home_schedule.get("fixtures") and away_schedule.get("fixtures"):
        score=int(clamp(score+4,0,100))
    level="High" if score>=80 else "Moderate" if score>=60 else "Limited"
    uncertainty=model_uncertainty(score,(home_profile,away_profile),matrix,fixture.get("raw_event",{}).get("provisional",False),team_news.get("home",{}).get("evidence_count",0)+team_news.get("away",{}).get("evidence_count",0))
    recommendation=best_bet_recommendation(value_rows,score,uncertainty,fixture.get("raw_event",{}).get("provisional",False))
    bet_gate = recommendation.get("decision")=="BEST VALUE OPTION"
    st.subheader("16. Model Safety Gate & Recommended Option")
    gate_label=recommendation.get("decision")
    rec_row=recommendation.get("row")
    if rec_row:
        detail=f"{rec_row['Outcome']} · model {pct(rec_row['Model Probability'])} · fair odds {rec_row['Fair Odds']:.2f} · offered odds {rec_row['Odds']:.2f} · edge {pct(rec_row['Edge'])}"
    else:
        detail=recommendation.get("reason","")
    st.markdown(f"<div class='decision'><b>{gate_label}</b><br>{detail}<br><span class='muted'>Coverage {score}/100 · uncertainty {uncertainty*100:.1f}% · top 1X2 probability {top_prob*100:.1f}%</span></div>",unsafe_allow_html=True)
    if gate_label=="BEST VALUE OPTION":
        st.success(f"MATCHLAB's strongest qualifying market is **{rec_row['Outcome']}**. This is the option the model would prioritize, subject to the bookmaker price you entered.")
    else:
        st.warning("MATCHLAB recommends **NO BET** under the current evidence/price. A prediction is not automatically a betting recommendation.")
    st.caption("Value requires both model probability and price. A high-probability outcome can still be a bad bet when the bookmaker price is too short.")

    st.subheader("17. Data Coverage")
    c1,c2,c3,c4=st.columns(4)
    with c1: st.metric("Coverage",f"{score}/100")
    with c2: st.metric("Level",level)
    with c3: st.metric("Recent matches",f"{home_profile['matches']} / {away_profile['matches']}")
    with c4: st.metric("Uncertainty",f"{uncertainty*100:.1f}%")

    model_source="TheSportsDB free structured data + public Internet research + MATCHLAB model"
    record={
        "fixture_id":fixture_id,
        "match_date":fixture["fixture"]["date"],
        "competition":competition,
        "home_team":home_team["name"],
        "away_team":away_team["name"],
        "home_probability":matrix["home_win"],
        "draw_probability":matrix["draw"],
        "away_probability":matrix["away_win"],
        "home_xg":home_xg,
        "away_xg":away_xg,
        "predicted_outcome":predicted_outcome,
        "predicted_score":f"{top_score[0]}-{top_score[1]}",
        "confidence_score":score,
        "bet_gate":bet_gate,
        "weather":weather_data,
        "motivation":motivation_data,
        "team_news":team_news,
        "future_fixtures":{"home":home_schedule,"away":away_schedule},
        "rotation_notes":{"home":home_rotation_note,"away":away_rotation_note},
        "uncertainty":uncertainty,
        "best_value_recommendation":recommendation,
        "value_rows":value_rows,
        "model_source":model_source,
        "value_summary":"Potential value detected" if value_rows and any(x["Potential Value"] for x in value_rows) else "No qualifying value detected",
        "reasons":reasons,
        "web_sources":[r.get("url") for r in web_research.get("results",[])[:15]],
    }

    if st.button("SAVE THIS PREDICTION",type="secondary",use_container_width=True):
        save_prediction(record)
        st.success("Prediction saved. Open Prediction History from the sidebar.")

    st.divider()
    st.markdown(f"### MATCHLAB final assessment: {predicted_outcome}")
    st.write(
        f"MATCHLAB gives {predicted_outcome} the highest model probability at "
        f"{pct(probs[predicted_outcome])}. The estimated scoring environment is "
        f"{home_xg:.2f}–{away_xg:.2f} xG. Future fixtures, rotation risk, team news, stadium weather and motivation are incorporated "
        "when reliable evidence exists, while bookmaker odds remain completely outside the prediction model until the separate value stage. "
        f"Current betting gate: {recommendation.get('decision')}."
    )
    st.warning(
        "This is a probabilistic model, not a guarantee. Free data coverage varies by competition, "
        "and some injury, lineup and statistical fields may be missing or delayed."
    )
