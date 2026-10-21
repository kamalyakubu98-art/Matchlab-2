
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
# MATCHLAB Pro V4
# Free public football data + live Internet research
# ============================================================

st.set_page_config(
    page_title="MATCHLAB Pro V4",
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

def find_fixture(home_name, away_name, match_date):
    wanted_date = match_date.isoformat()

    # Fast exact event-title search.
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
                    return event_to_fixture(event), event_to_fixture(event)["teams"]["home"], event_to_fixture(event)["teams"]["away"]
        except Exception:
            continue

    # Exact day search fallback.
    try:
        events = football.events_day(wanted_date)
        for event in events:
            if event_matches_fixture(event, home_name, away_name):
                fixture = event_to_fixture(event)
                return fixture, fixture["teams"]["home"], fixture["teams"]["away"]
    except Exception:
        pass

    raise RuntimeError(
        "The free football source could not verify this exact fixture. "
        "Try adding a league hint or check the spelling/date."
    )

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
    results = []
    seen = set()
    for searcher in (ddg_search, google_news_search):
        try:
            found = searcher(query, max_results=max_results)
        except Exception:
            found = []
        for item in found:
            key = item.get("url") or item.get("title")
            if key and key not in seen:
                seen.add(key)
                results.append(item)
            if len(results) >= max_results:
                return results
    return results

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
# ODDS INPUT — USER SUPPLIED, NEVER A MODEL INPUT
# ============================================================

def user_odds_value_analysis(matrix_result, home_team, away_team, home_odds, draw_odds, away_odds):
    odds = {
        home_team: safe_float(home_odds),
        "Draw": safe_float(draw_odds),
        away_team: safe_float(away_odds),
    }
    valid = {k: v for k, v in odds.items() if v and v > 1.0}
    if not valid:
        return None

    raw = {k: 1.0 / v for k, v in valid.items()}
    total = sum(raw.values())
    market = {k: v / total for k, v in raw.items()}

    model = {
        home_team: matrix_result["home_win"],
        "Draw": matrix_result["draw"],
        away_team: matrix_result["away_win"],
    }

    rows = []
    for outcome, mp in model.items():
        if outcome not in market:
            continue
        edge = mp - market[outcome]
        fair = fair_odds(mp)
        rows.append({
            "Outcome": outcome,
            "Model Probability": mp,
            "Market Probability": market[outcome],
            "Edge": edge,
            "Odds": valid[outcome],
            "Fair Odds": fair,
            "Potential Value": bool(edge >= 0.05 and valid[outcome] > fair),
        })
    return rows

# ============================================================
# HISTORICAL / RECENT TEAM MODEL
# ============================================================

def completed_fi
