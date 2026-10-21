import json
import math
import os
import random
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from collections import Counter

import pandas as pd
import requests
import streamlit as st

# ============================================================
# MATCHLAB Pro V3
# Fixture-first, data-driven football prediction engine
# ============================================================

st.set_page_config(
    page_title="MATCHLAB Pro V3",
    page_icon="⚽",
    layout="wide",
)

DB_PATH = Path("matchlab_history.db")
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"

# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------

st.markdown(
    """
    <style>
    .title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0;
    }
    .subtitle {
        color: #8f9aaa;
        margin-bottom: 18px;
    }
    .welcome {
        padding: 18px;
        border-radius: 14px;
        background: rgba(80, 100, 130, 0.12);
        border: 1px solid rgba(130, 150, 180, 0.20);
        margin-bottom: 20px;
    }
    .reason {
        padding: 14px;
        border-radius: 12px;
        margin-bottom: 10px;
        background: rgba(80, 100, 130, 0.08);
        border: 1px solid rgba(130, 150, 180, 0.14);
    }
    .value-box {
        padding: 16px;
        border-radius: 14px;
        background: rgba(30, 140, 80, 0.10);
        border: 1px solid rgba(30, 160, 90, 0.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SETTINGS / SECRETS
# ============================================================

def get_secret(name):
    try:
        value = st.secrets.get(name, "")
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, "")


API_FOOTBALL_KEY = get_secret("API_FOOTBALL_KEY")
TAVILY_API_KEY = get_secret("TAVILY_API_KEY")
SERPER_API_KEY = get_secret("SERPER_API_KEY")

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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                fixture_id INTEGER,
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
            """
        )
        conn.commit()


init_db()


def save_prediction(record):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO predictions (
                created_at, fixture_id, match_date, competition,
                home_team, away_team,
                home_probability, draw_probability, away_probability,
                home_xg, away_xg,
                predicted_outcome, predicted_score,
                confidence_score, model_source,
                value_summary, reasons_json, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(),
                record.get("fixture_id"),
                record.get("match_date"),
                record.get("competition"),
                record["home_team"],
                record["away_team"],
                record["home_probability"],
                record["draw_probability"],
                record["away_probability"],
                record["home_xg"],
                record["away_xg"],
                record["predicted_outcome"],
                record["predicted_score"],
                record.get("confidence_score"),
                record.get("model_source"),
                record.get("value_summary", ""),
                json.dumps(record.get("reasons", [])),
                json.dumps(record, default=str),
            ),
        )
        conn.commit()


def load_history(limit=100):
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            """
            SELECT id, created_at, fixture_id, match_date,
                   competition, home_team, away_team,
                   home_probability, draw_probability,
                   away_probability, home_xg, away_xg,
                   predicted_outcome, predicted_score,
                   confidence_score, model_source,
                   actual_home_goals, actual_away_goals,
                   result_checked, value_summary
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )


def delete_prediction(prediction_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "DELETE FROM predictions WHERE id = ?",
            (int(prediction_id),),
        )
        conn.commit()


def update_actual_result(prediction_id, home_goals, away_goals):
    actual_result = (
        "Home"
        if home_goals > away_goals
        else "Draw"
        if home_goals == away_goals
        else "Away"
    )

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            UPDATE predictions
            SET actual_home_goals = ?,
                actual_away_goals = ?,
                result_checked = 1
            WHERE id = ?
            """,
            (home_goals, away_goals, int(prediction_id)),
        )
        conn.commit()

# ============================================================
# API-FOOTBALL CLIENT
# ============================================================

class APIFootball:
    def __init__(self, api_key):
        self.api_key = api_key

    def enabled(self):
        return bool(self.api_key)

    def get(self, endpoint, params=None, timeout=20):
        if not self.enabled():
            raise RuntimeError("API_FOOTBALL_KEY is not configured.")

        response = requests.get(
            API_FOOTBALL_BASE + endpoint,
            headers={
                "x-apisports-key": self.api_key,
                "Accept": "application/json",
            },
            params=params or {},
            timeout=timeout,
        )

        response.raise_for_status()
        payload = response.json()

        if payload.get("errors"):
            raise RuntimeError(str(payload["errors"]))

        return payload

    def search_team(self, name):
        data = self.get("/teams", {"search": name})
        return data.get("response", [])

    def team_fixtures(self, team_id, last=None, next_count=None):
        params = {"team": team_id}
        if last is not None:
            params["last"] = last
        if next_count is not None:
            params["next"] = next_count
        data = self.get("/fixtures", params)
        return data.get("response", [])

    def fixture_by_id(self, fixture_id):
        data = self.get("/fixtures", {"id": fixture_id})
        response = data.get("response", [])
        return response[0] if response else None

    def head_to_head(self, home_id, away_id, last=10):
        data = self.get(
            "/fixtures/headtohead",
            {"h2h": f"{home_id}-{away_id}", "last": last},
        )
        return data.get("response", [])

    def injuries(self, fixture_id=None, team_id=None):
        params = {}
        if fixture_id:
            params["fixture"] = fixture_id
        elif team_id:
            params["team"] = team_id
        data = self.get("/injuries", params)
        return data.get("response", [])

    def lineups(self, fixture_id):
        data = self.get("/fixtures/lineups", {"fixture": fixture_id})
        return data.get("response", [])

    def odds(self, fixture_id):
        data = self.get("/odds", {"fixture": fixture_id})
        return data.get("response", [])

    def standings(self, league_id, season):
        data = self.get(
            "/standings",
            {"league": league_id, "season": season},
        )
        return data.get("response", [])

api = APIFootball(API_FOOTBALL_KEY)

# ============================================================
# OPTIONAL GENERAL WEB RESEARCH
# ============================================================

def web_search(query, max_results=5):
    """Use Tavily first, then Serper if configured."""

    if TAVILY_API_KEY:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": "advanced",
                "max_results": max_results,
                "include_answer": True,
            },
            timeout=25,
        )
        response.raise_for_status()
        payload = response.json()
        return {
            "provider": "Tavily",
            "answer": payload.get("answer", ""),
            "results": payload.get("results", []),
        }

    if SERPER_API_KEY:
        response = requests.post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": SERPER_API_KEY,
                "Content-Type": "application/json",
            },
            json={"q": query, "num": max_results},
            timeout=25,
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("organic", [])
        return {
            "provider": "Serper",
            "answer": "",
            "results": results,
        }

    return None

# ============================================================
# TEAM / FIXTURE IDENTIFICATION
# ============================================================

def choose_team(candidates, requested):
    if not candidates:
        return None

    requested_norm = norm(requested)

    exact = [
        item for item in candidates
        if norm(item.get("team", {}).get("name")) == requested_norm
    ]

    if exact:
        return exact[0]

    return candidates[0]


def find_fixture(home_name, away_name, match_date):
    home_candidates = api.search_team(home_name)
    away_candidates = api.search_team(away_name)

    home_match = choose_team(home_candidates, home_name)
    away_match = choose_team(away_candidates, away_name)

    if not home_match or not away_match:
        raise RuntimeError(
            "One or both teams could not be identified by the football data provider."
        )

    home_team = home_match["team"]
    away_team = away_match["team"]

    fixtures = api.team_fixtures(
        home_team["id"],
        next_count=30,
    )

    wanted = []

    for fixture in fixtures:
        f_date = fixture.get("fixture", {}).get("date", "")[:10]
        f_home = fixture.get("teams", {}).get("home", {}).get("id")
        f_away = fixture.get("teams", {}).get("away", {}).get("id")

        if (
            f_date == match_date.isoformat()
            and f_home == home_team["id"]
            and f_away == away_team["id"]
        ):
            wanted.append(fixture)

    if wanted:
        return wanted[0], home_team, away_team

    # Fallback: inspect the away team's next fixtures too.
    away_fixtures = api.team_fixtures(
        away_team["id"],
        next_count=30,
    )

    for fixture in away_fixtures:
        f_date = fixture.get("fixture", {}).get("date", "")[:10]
        f_home = fixture.get("teams", {}).get("home", {}).get("id")
        f_away = fixture.get("teams", {}).get("away", {}).get("id")

        if (
            f_date == match_date.isoformat()
            and f_home == home_team["id"]
            and f_away == away_team["id"]
        ):
            return fixture, home_team, away_team

    raise RuntimeError(
        "No exact fixture was found for those teams on the selected date. "
        "Check the spelling, date, or fixture schedule."
    )

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

    if second_leg:
        if aggregate_for < aggregate_against:
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
    st.header("MATCHLAB V3")

    page = st.radio(
        "Workspace",
        [
            "Analyze Fixture",
            "Prediction History",
        ],
    )

    st.divider()

    st.subheader("Data Connections")

    if API_FOOTBALL_KEY:
        st.success("Football data API connected")
    else:
        st.error("API-Football key missing")

    if TAVILY_API_KEY or SERPER_API_KEY:
        st.success("Web research connected")
    else:
        st.warning("Optional web research not connected")

    st.caption(
        "Add API_FOOTBALL_KEY and optionally TAVILY_API_KEY or SERPER_API_KEY "
        "to Streamlit Secrets."
    )

# ============================================================
# HISTORY PAGE
# ============================================================

if page == "Prediction History":
    st.markdown(
        '<div class="title">MATCHLAB History</div>',
        unsafe_allow_html=True,
    )
    st.caption("Previous predictions saved by this MATCHLAB instance.")

    history_df = load_history(200)

    if history_df.empty:
        st.info("No previous predictions have been saved yet.")
        st.stop()

    display_df = history_df.copy()

    display_df["Prediction"] = display_df.apply(
        lambda r: (
            f"{r['predicted_outcome']} "
            f"({r['predicted_score']})"
        ),
        axis=1,
    )

    display_df["Model"] = display_df[
        ["home_probability", "draw_probability", "away_probability"]
    ].max(axis=1).map(lambda x: f"{x * 100:.1f}%")
                        columns = [
        "id",
        "created_at",
        "match_date",
        "competition",
        "home_team",
        "away_team",
        "Prediction",
        "Model",
        "confidence_score",
        "value_summary",
        "result_checked",
    ]

    st.dataframe(
        display_df[columns],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Check a Previous Prediction")

    selected_id = st.selectbox(
        "Prediction ID",
        display_df["id"].tolist(),
    )

    selected = display_df[
        display_df["id"] == selected_id
    ].iloc[0]

    st.markdown(
        f"### {selected['home_team']} vs {selected['away_team']}"
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Home",
            pct(selected["home_probability"]),
        )

    with c2:
        st.metric(
            "Draw",
            pct(selected["draw_probability"]),
        )

    with c3:
        st.metric(
            "Away",
            pct(selected["away_probability"]),
        )

    if int(selected["result_checked"] or 0) == 1:
        actual = (
            f"{int(selected['actual_home_goals'])}-"
            f"{int(selected['actual_away_goals'])}"
        )
        st.success(f"Actual result recorded: {actual}")
    else:
        st.info("Actual result has not been recorded yet.")

    with st.expander("Record actual result"):
        actual_home = st.number_input(
            "Actual home goals",
            min_value=0,
            max_value=20,
            value=0,
            key="history_actual_home",
        )
        actual_away = st.number_input(
            "Actual away goals",
            min_value=0,
            max_value=20,
            value=0,
            key="history_actual_away",
        )

        if st.button("Save Actual Result"):
            update_actual_result(
                selected_id,
                actual_home,
                actual_away,
            )
            st.success("Actual result saved.")
            st.rerun()

    if st.button("Delete Selected Prediction"):
        delete_prediction(selected_id)
        st.success("Prediction deleted.")
        st.rerun()

    st.stop()

# ============================================================
# MAIN ANALYSIS PAGE
# ============================================================

st.markdown(
    '<div class="title">MATCHLAB Pro V3</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">Fixture-first football analysis · data before odds · value after prediction</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="welcome">
    <b>Enter any real fixture.</b> MATCHLAB identifies the fixture,
    gathers available football data, builds its own probability model,
    explains the result and only then checks optional market odds for value.
    </div>
    """,
    unsafe_allow_html=True,
)

if not API_FOOTBALL_KEY:
    st.error(
        "MATCHLAB cannot perform live fixture analysis until API_FOOTBALL_KEY "
        "is added to Streamlit Secrets."
    )
    st.markdown(
        "Use a Streamlit secret named `API_FOOTBALL_KEY`. "
        "The API provides fixtures, team data, head-to-head, injuries, lineups and odds."
    )
    st.stop()
# ------------------------------------------------------------
# Fixture input
# ------------------------------------------------------------

st.subheader("1. Real-World Fixture")

f1, f2 = st.columns(2)

with f1:
    home_team_input = st.text_input(
        "Home team",
        placeholder="e.g. Arsenal",
    )

with f2:
    away_team_input = st.text_input(
        "Away team",
        placeholder="e.g. Chelsea",
    )

f3, f4 = st.columns(2)

with f3:
    match_date = st.date_input(
        "Match date",
        value=date.today() + timedelta(days=1),
    )

with f4:
    competition_hint = st.text_input(
        "Competition / league hint (optional)",
        placeholder="e.g. Premier League",
    )

advanced_col1, advanced_col2 = st.columns(2)

with advanced_col1:
    max_goals = st.slider(
        "Maximum score matrix goals",
        5,
        12,
        9,
    )

with advanced_col2:
    simulations = st.selectbox(
        "Monte Carlo simulations",
        [5000, 10000, 25000],
        index=1,
    )

# ------------------------------------------------------------
# Run analysis
# ------------------------------------------------------------

analyze = st.button(
    "ANALYZE FIXTURE",
    type="primary",
    use_container_width=True,
)

if analyze:
    if not home_team_input.strip() or not away_team_input.strip():
        st.error("Enter both teams.")
        st.stop()

    if norm(home_team_input) == norm(away_team_input):
        st.error("Home and away teams must be different.")
        st.stop()

    with st.spinner("Identifying fixture and collecting football data..."):
        try:
            fixture, home_team, away_team = find_fixture(
                home_team_input.strip(),
                away_team_input.strip(),
                match_date,
            )
        except Exception as error:
            st.error(f"Fixture identification failed: {error}")
            st.stop()

        fixture_id = fixture.get("fixture", {}).get("id")

        try:
            home_fixtures = api.team_fixtures(
                home_team["id"],
                last=15,
            )
            away_fixtures = api.team_fixtures(
                away_team["id"],
                last=15,
            )

            home_profile = build_recent_team_profile(
                home_fixtures,
                home_team["id"],
            )
            away_profile = build_recent_team_profile(
                away_fixtures,
                away_team["id"],
            )

            h2h = api.head_to_head(
                home_team["id"],
                away_team["id"],
                last=10,
            )

            injuries_home = api.injuries(
                fixture_id=fixture_id,
            )
            injuries_away = injuries_home

            # Split fixture injury response by team.
            injuries_home = [
                x for x in injuries_home
                if x.get("team", {}).get("id") == home_team["id"]
            ]
            injuries_away = [
                x for x in api.injuries(fixture_id=fixture_id)
                if x.get("team", {}).get("id") == away_team["id"]
            ]

            try:
                lineups = api.lineups(fixture_id)
            except Exception:
                lineups = []

        except Exception as error:
            st.error(f"Data collection failed: {error}")
            st.stop()

    # --------------------------------------------------------
    # Optional web research
    # --------------------------------------------------------
web_research = None

    if TAVILY_API_KEY or SERPER_API_KEY:
        with st.spinner("Checking relevant web information..."):
            query = (
                f"{home_team['name']} vs {away_team['name']} "
                f"{match_date.isoformat()} football injuries lineup news"
            )
            try:
                web_research = web_search(query, max_results=5)
            except Exception as error:
                st.warning(f"Web research was unavailable: {error}")

    # --------------------------------------------------------
    # Infer venue
    # --------------------------------------------------------

    venue_name = (
        fixture.get("fixture", {})
        .get("venue", {})
        .get("name")
        or "Unknown venue"
    )

    # --------------------------------------------------------
    # Context from actual fixture
    # --------------------------------------------------------

    home_context = context_multiplier(
        formation="4-3-3",
        motivation="Normal",
        absences=len(injuries_home),
        absence_type="Mixed",
        absence_importance=50,
        profile=home_profile,
        rest_days=5,
        aggregate_for=0,
        aggregate_against=0,
        second_leg=False,
    )

    away_context = context_multiplier(
        formation="4-3-3",
        motivation="Normal",
        absences=len(injuries_away),
        absence_type="Mixed",
        absence_importance=50,
        profile=away_profile,
        rest_days=5,
        aggregate_for=0,
        aggregate_against=0,
        second_leg=False,
    )

    home_xg, away_xg = predict_xg(
        home_profile,
        away_profile,
        home_context,
        away_context,
        "Home team venue",
        "Clear",
        "Good",
    )

    matrix = calculate_matrix(
        home_xg,
        away_xg,
        max_goals=max_goals,
        rho=-0.05,
    )

    simulation = run_monte_carlo(
        home_xg,
        away_xg,
        simulations,
        seed=42,
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    st.success(
        f"Fixture identified: {home_team['name']} vs {away_team['name']}"
    )

    meta1, meta2, meta3 = st.columns(3)

    with meta1:
        st.metric(
            "Competition",
            fixture.get("league", {}).get("name", "Unknown"),
        )

    with meta2:
        st.metric(
            "Venue",
            venue_name,
        )

    with meta3:
        st.metric(
            "Fixture ID",
            str(fixture_id),
        )

    # --------------------------------------------------------
    # Main prediction
    # --------------------------------------------------------

    st.subheader("2. MATCHLAB Prediction")

    probs = {
        home_team["name"]: matrix["home_win"],
        "Draw": matrix["draw"],
        away_team["name"]: matrix["away_win"],
    }

    predicted_outcome = max(
        probs,
        key=probs.get,
    )

    top_score = matrix["top_scores"][0]

    p1, p2, p3 = st.columns(3)

    with p1:
        st.metric(
            home_team["name"],
            pct(matrix["home_win"]),
        )

    with p2:
        st.metric(
            "Draw",
            pct(matrix["draw"]),
        )

    with p3:
        st.metric(
            away_team["name"],
            pct(matrix["away_win"]),
        )

    x1, x2, x3 = st.columns(3)

    with x1:
        st.metric(
            f"{home_team['name']} xG",
            f"{home_xg:.2f}",
        )

    with x2:
        st.metric(
            "Total xG",
            f"{home_xg + away_xg:.2f}",
        )

    with x3:
        st.metric(
            f"{away_team['name']} xG",
            f"{away_xg:.2f}",
        )

    st.markdown(
        f"### Most likely outcome: **{predicted_outcome}**"
    )

    st.markdown(
        f"Most likely exact score: **{top_score[0]}-{top_score[1]}** "
        f"({top_score[2] * 100:.2f}%)"
    )
    # --------------------------------------------------------
    # Why MATCHLAB produced the result
    # --------------------------------------------------------

    reasons, risks = build_explanation(
        home_team["name"],
        away_team["name"],
        home_profile,
        away_profile,
        home_xg,
        away_xg,
        matrix,
        injuries_home,
        injuries_away,
        h2h,
        home_context,
        away_context,
    )

    st.subheader("3. Why MATCHLAB Produced This Result")

    for index, reason in enumerate(reasons, start=1):
        st.markdown(
            f"<div class='reason'><b>{index}.</b> {reason}</div>",
            unsafe_allow_html=True,
        )

    st.subheader("4. Why Other Outcomes Could Still Happen")

    for index, risk in enumerate(risks, start=1):
        st.markdown(
            f"<div class='reason'><b>{index}.</b> {risk}</div>",
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # Recent form evidence
    # --------------------------------------------------------

    st.subheader("5. Recent Evidence")

    evidence_df = pd.DataFrame(
        [
            {
                "Metric": "Recent matches",
                home_team["name"]: home_profile["matches"],
                away_team["name"]: away_profile["matches"],
            },
            {
                "Metric": "Weighted goals scored",
                home_team["name"]: round(home_profile["gf"], 2),
                away_team["name"]: round(away_profile["gf"], 2),
            },
            {
                "Metric": "Weighted goals conceded",
                home_team["name"]: round(home_profile["ga"], 2),
                away_team["name"]: round(away_profile["ga"], 2),
            },
            {
                "Metric": "Weighted points per match",
                home_team["name"]: round(home_profile["points"], 2),
                away_team["name"]: round(away_profile["points"], 2),
            },
            {
                "Metric": "Home/Away scoring rate",
                home_team["name"]: round(home_profile["home_gf"], 2),
                away_team["name"]: round(away_profile["away_gf"], 2),
            },
            {
                "Metric": "Home/Away conceded rate",
                home_team["name"]: round(home_profile["home_ga"], 2),
                away_team["name"]: round(away_profile["away_ga"], 2),
            },
        ]
    )

    st.dataframe(
        evidence_df,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # Injury / lineup evidence
    # --------------------------------------------------------

    st.subheader("6. Availability")

    availability_rows = []

    for item in injuries_home:
        player = item.get("player", {}).get("name", "Unknown")
        reason = item.get("player", {}).get("reason", "Unknown")
        availability_rows.append({
            "Team": home_team["name"],
            "Player": player,
            "Type": item.get("type", ""),
            "Reason": reason,
        })

    for item in injuries_away:
        player = item.get("player", {}).get("name", "Unknown")
        reason = item.get("player", {}).get("reason", "Unknown")
        availability_rows.append({
            "Team": away_team["name"],
            "Player": player,
            "Type": item.get("type", ""),
            "Reason": reason,
        })

    if availability_rows:
        st.dataframe(
            pd.DataFrame(availability_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No fixture-specific availability issues were returned by the provider.")

    if lineups:
        lineup_rows = []
        for block in lineups:
            team = block.get("team", {}).get("name", "Unknown")
            formation = block.get("formation") or "Unknown"
            lineup_rows.append({
                "Team": team,
                "Formation": formation,
                "Starting XI": len(block.get("startXI", [])),
                "Substitutes": len(block.get("substitutes", [])),
            })

        st.dataframe(
            pd.DataFrame(lineup_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("Confirmed lineups are not yet available or not covered for this fixture.")
# --------------------------------------------------------
    # Goal markets
    # --------------------------------------------------------

    st.subheader("8. Goal Markets")

    market_df = pd.DataFrame(
        [
            ["Over 1.5", pct(matrix["over_15"])],
            ["Under 1.5", pct(matrix["under_15"])],
            ["Over 2.5", pct(matrix["over_25"])],
            ["Under 2.5", pct(matrix["under_25"])],
            ["Over 3.5", pct(matrix["over_35"])],
            ["Under 3.5", pct(matrix["under_35"])],
            ["BTTS Yes", pct(matrix["btts_yes"])],
            ["BTTS No", pct(matrix["btts_no"])],
            ["Home Clean Sheet", pct(matrix["home_clean_sheet"])],
            ["Away Clean Sheet", pct(matrix["away_clean_sheet"])],
        ],
        columns=["Market", "Probability"],
    )

    st.dataframe(
        market_df,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # Correct scores
    # --------------------------------------------------------

    st.subheader("9. Correct Score Distribution")

    score_df = pd.DataFrame(
        [
            {
                "Score": f"{hg}-{ag}",
                "Probability": pct(probability),
            }
            for hg, ag, probability in matrix["top_scores"][:10]
        ]
    )

    st.dataframe(
        score_df,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # Optional web research report
    # --------------------------------------------------------

    if web_research:
        st.subheader("10. Internet Research")

        if web_research.get("answer"):
            st.write(web_research["answer"])

        for result in web_research.get("results", [])[:5]:
            title = result.get("title", "Source")
            url = result.get("url", "")
            content = result.get("content", result.get("snippet", ""))

            if url:
                st.markdown(f"**{title}**")
                st.write(content[:700])
                st.caption(url)

        st.caption(
            "Web research is supporting evidence. It does not directly overwrite the probability model."
        )

    # --------------------------------------------------------
    # Odds AFTER model
    # --------------------------------------------------------

    st.subheader("11. Optional Market Value Check")

    st.caption(
        "Odds are deliberately checked only after MATCHLAB has produced its own probabilities. "
        "They do not feed back into the prediction."
    )

    odds_rows = []

    try:
        odds_response = api.odds(fixture_id)
        odds_rows = parse_odds_response(odds_response)
    except Exception as error:
        st.caption(f"Market odds unavailable: {error}")

    value_rows = value_analysis(
        matrix,
        odds_rows,
        home_team["name"],
        away_team["name"],
    )

    if value_rows:
        value_df = pd.DataFrame(value_rows)
        value_df["Model Probability"] = value_df[
            "Model Probability"
        ].map(pct)
        value_df["Market Probability"] = value_df[
            "Market Probability"
        ].map(pct)
        value_df["Edge"] = value_df["Edge"].map(pct)
        value_df["Model Fair Odds"] = value_df[
            "Model Fair Odds"
        ].map(lambda x: f"{x:.2f}")
        value_df["Best Odds"] = value_df[
            "Best Odds"
        ].map(lambda x: f"{x:.2f}" if x else "-")

        st.dataframe(
            value_df,
            use_container_width=True,
            hide_index=True,
        )

        positives = [
            row for row in value_rows
            if row["Potential Value"]
        ]

        if positives:
            for row in positives:
                st.markdown(
                    f"<div class='value-box'><b>Potential value:</b> "
                    f"{row['Outcome']} · model {pct(row['Model Probability'])} · "
                    f"market {pct(row['Market Probability'])} · "
                    f"edge {pct(row['Edge'])} · "
                    f"odds {row['Best Odds']:.2f}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No qualifying model-vs-market edge was detected under the current threshold.")

    else:
        st.info(
            "No usable 1X2 bookmaker odds were returned. "
            "The prediction remains independent of market prices."
    )
        # --------------------------------------------------------
    # Confidence / coverage
    # --------------------------------------------------------

    score, level = coverage_score(
        home_profile,
        away_profile,
        fixture,
        injuries_home,
        injuries_away,
        h2h,
    )

    st.subheader("12. Data Coverage")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Coverage", f"{score}/100")

    with c2:
        st.metric("Level", level)

    with c3:
        st.metric("Historical sample", f"{home_profile['matches'] + away_profile['matches']}")

    st.caption(
        "Coverage is a measure of how much relevant information MATCHLAB obtained; "
        "it is not a guarantee that the prediction will be correct."
    )

    # --------------------------------------------------------
    # Save prediction
    # --------------------------------------------------------

    model_source = "API-Football structured data + MATCHLAB model"

    record = {
        "fixture_id": fixture_id,
        "match_date": fixture.get("fixture", {}).get("date"),
        "competition": fixture.get("league", {}).get("name", ""),
        "home_team": home_team["name"],
        "away_team": away_team["name"],
        "home_probability": matrix["home_win"],
        "draw_probability": matrix["draw"],
        "away_probability": matrix["away_win"],
        "home_xg": home_xg,
        "away_xg": away_xg,
        "predicted_outcome": predicted_outcome,
        "predicted_score": f"{top_score[0]}-{top_score[1]}",
        "confidence_score": score,
        "model_source": model_source,
        "value_summary": (
            "Potential value detected"
            if value_rows and any(x["Potential Value"] for x in value_rows)
            else "No qualifying value detected"
        ),
        "reasons": reasons,
    }

    if st.button("SAVE THIS PREDICTION", type="secondary", use_container_width=True):
        save_prediction(record)
        st.success(
            "Prediction saved. Open Prediction History from the sidebar to review it later."
        )

    # --------------------------------------------------------
    # Final model statement
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        f"### MATCHLAB final assessment: {predicted_outcome}"
    )

    st.write(
        f"MATCHLAB gives {predicted_outcome} the highest model probability "
        f"at {pct(probs[predicted_outcome])}. The model's estimated scoring environment "
        f"is {home_xg:.2f}–{away_xg:.2f} xG. The main reasons above explain the direction; "
        f"the counterarguments explain why the other outcomes remain possible."
    )

    st.warning(
        "This is a probabilistic football model, not a guarantee. Data coverage varies by competition, "
        "and some injury, lineup and statistical fields may be missing or delayed."
    )
    
  
