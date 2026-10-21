import io
import json
import math
import random
from collections import Counter, defaultdict

import pandas as pd
import streamlit as st


# ============================================================
# MATCHLAB PRO V2
# Context-aware football probability + simulation workspace
# ============================================================

st.set_page_config(
    page_title="MATCHLAB Pro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 3rem;
            font-weight: 900;
            letter-spacing: -2px;
            margin-bottom: 0;
        }

        .subtitle {
            color: #777;
            font-size: 1.05rem;
            margin-top: -5px;
            margin-bottom: 25px;
        }

        .welcome {
            padding: 22px;
            border-radius: 14px;
            background: linear-gradient(
                135deg,
                rgba(30, 100, 220, 0.10),
                rgba(30, 180, 120, 0.08)
            );
            border: 1px solid rgba(100, 100, 100, 0.15);
            margin-bottom: 20px;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 800;
        }

        .small-note {
            color: #777;
            font-size: 0.82rem;
        }

        .section-card {
            padding: 18px;
            border-radius: 12px;
            border: 1px solid rgba(100, 100, 100, 0.15);
            margin-bottom: 15px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# BUILT-IN TEAM PRIORS
# These are fallback/illustrative ratings.
# ============================================================

TEAM_DATA = {
    "aston villa": {
        "attack": 1.65,
        "defense": 1.20,
        "home_adv": 0.25,
    },
    "manchester city": {
        "attack": 2.30,
        "defense": 0.90,
        "home_adv": 0.30,
    },
    "arsenal": {
        "attack": 2.10,
        "defense": 0.85,
        "home_adv": 0.22,
    },
    "liverpool": {
        "attack": 2.25,
        "defense": 0.95,
        "home_adv": 0.35,
    },
    "chelsea": {
        "attack": 1.80,
        "defense": 1.35,
        "home_adv": 0.15,
    },
    "manchester united": {
        "attack": 1.45,
        "defense": 1.40,
        "home_adv": 0.10,
    },
    "real madrid": {
        "attack": 2.40,
        "defense": 0.88,
        "home_adv": 0.32,
    },
    "barcelona": {
        "attack": 2.05,
        "defense": 1.05,
        "home_adv": 0.28,
    },
    "bayern munich": {
        "attack": 2.35,
        "defense": 1.10,
        "home_adv": 0.30,
    },
    "inter milan": {
        "attack": 1.95,
        "defense": 0.80,
        "home_adv": 0.25,
    },
    "juventus": {
        "attack": 1.60,
        "defense": 0.90,
        "home_adv": 0.18,
    },
    "psg": {
        "attack": 2.15,
        "defense": 1.15,
        "home_adv": 0.20,
    },
}


ALIASES = {
    "man city": "manchester city",
    "mancity": "manchester city",
    "manchester utd": "manchester united",
    "man utd": "manchester united",
    "man united": "manchester united",
    "real": "real madrid",
    "real madrid cf": "real madrid",
    "barca": "barcelona",
    "fc barcelona": "barcelona",
    "bayern": "bayern munich",
    "fc bayern": "bayern munich",
    "inter": "inter milan",
    "internazionale": "inter milan",
    "paris saint-germain": "psg",
    "paris sg": "psg",
}


FORMATIONS = [
    "4-3-3",
    "4-2-3-1",
    "4-4-2",
    "3-4-3",
    "3-5-2",
    "5-3-2",
    "4-1-4-1",
    "5-4-1",
    "Not specified",
]


COMPETITIONS = [
    "Premier League",
    "UEFA Champions League",
    "UEFA Europa League",
    "La Liga",
    "Bundesliga",
    "Serie A",
    "Ligue 1",
    "Domestic Cup",
    "International",
    "Friendly",
    "Other",
]


MATCH_STAGES = [
    "League / regular match",
    "Group stage",
    "Knockout — first leg",
    "Knockout — second leg",
    "Final",
    "Friendly",
]


VENUES = [
    "Home team venue",
    "Away team venue",
    "Neutral venue",
    "Not sure",
]


ABSENCE_TYPES = [
    "Not specified",
    "Attackers",
    "Midfielders",
    "Defenders",
    "Goalkeeper",
    "Mixed",
]


MOTIVATIONS = [
    "Normal",
    "Low",
    "High",
    "Very high",
]


# ============================================================
# BASIC HELPERS
# ============================================================

def norm(name):
    """Normalize team names and apply aliases."""
    if name is None:
        return ""

    value = str(name).strip().lower()
    value = " ".join(value.split())

    return ALIASES.get(value, value)


def clamp(value, low, high):
    return max(low, min(high, value))


def poisson_pmf(k, lam):
    """Poisson probability."""
    lam = max(0.0001, float(lam))

    if k < 0:
        return 0.0

    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def weighted_mean(values, weights):
    total_weight = sum(weights)

    if total_weight <= 0:
        return float(sum(values) / max(1, len(values)))

    return sum(v * w for v, w in zip(values, weights)) / total_weight


# ============================================================
# FORM PARSER
# ============================================================

def parse_form(text):
    """
    Accepts:
        2-1, 1-1, 3-0
    or:
        2-1; 1-1; 3-0
    """
    if not text or not text.strip():
        return []

    entries = []

    chunks = text.replace(";", ",").split(",")

    for chunk in chunks:
        chunk = chunk.strip()

        if not chunk:
            continue

        if "-" not in chunk:
            raise ValueError(
                f"Invalid form entry '{chunk}'. Use format GF-GA, e.g. 2-1."
            )

        left, right = chunk.split("-", 1)

        try:
            gf = int(left.strip())
            ga = int(right.strip())
        except ValueError:
            raise ValueError(
                f"Invalid form entry '{chunk}'. Use numbers such as 2-1."
            )

        if not (0 <= gf <= 20 and 0 <= ga <= 20):
            raise ValueError(
                f"Goals in '{chunk}' must be between 0 and 20."
            )

        entries.append((gf, ga))

    return entries[-10:]


def form_adjustment(entries, baseline_goals=1.35):
    """
    Returns:
        attack_multiplier
        defense_multiplier
        summary

    Defense multiplier:
        < 1 = better defensive form
        > 1 = worse defensive form
    """

    if not entries:
        return 1.0, 1.0, None

    n = len(entries)

    # Most recent match receives the largest weight.
    weights = [
        0.72 ** (n - 1 - i)
        for i in range(n)
    ]

    gf_values = [x[0] for x in entries]
    ga_values = [x[1] for x in entries]

    points = []

    for gf, ga in entries:
        if gf > ga:
            points.append(3)
        elif gf == ga:
            points.append(1)
        else:
            points.append(0)

    gf_avg = weighted_mean(gf_values, weights)
    ga_avg = weighted_mean(ga_values, weights)
    points_avg = weighted_mean(points, weights)

    attack_multiplier = (
        1
        + (gf_avg - baseline_goals) * 0.06
        + (points_avg - 1.35) * 0.02
    )

    defense_multiplier = (
        1
        + (ga_avg - baseline_goals) * 0.05
    )

    attack_multiplier = clamp(
        attack_multiplier,
        0.90,
        1.10,
    )

    defense_multiplier = clamp(
        defense_multiplier,
        0.92,
        1.08,
    )

    summary = {
        "matches": n,
        "gf_avg": gf_avg,
        "ga_avg": ga_avg,
        "points_avg": points_avg,
    }

    return attack_multiplier, defense_multiplier, summary


# ============================================================
# HISTORICAL CSV SUPPORT
# ============================================================

def column_key(value):
    """
    Turns:
        Home Team
        home_team
        HomeTeam

    into the same comparable key.
    """
    return "".join(
        character
        for character in str(value).lower()
        if character.isalnum()
    )


def find_column(columns, candidates):
    lookup = {
        column_key(column): column
        for column in columns
    }

    for candidate in candidates:
        key = column_key(candidate)

        if key in lookup:
            return lookup[key]

    return None


def normalize_history_dataframe(df):
    """
    Accepts many common football CSV column names.

    Required:
        home_team
        away_team
        home_goals
        away_goals

    Optional:
        date
        home_xg
        away_xg
        competition
    """

    home_team_col = find_column(
        df.columns,
        [
            "home_team",
            "home",
            "hometeam",
            "home_team_name",
        ],
    )

    away_team_col = find_column(
        df.columns,
        [
            "away_team",
            "away",
            "awayteam",
            "away_team_name",
        ],
    )

    home_goals_col = find_column(
        df.columns,
        [
            "home_goals",
            "home_score",
            "fthg",
            "hg",
            "homegoals",
        ],
    )

    away_goals_col = find_column(
        df.columns,
        [
            "away_goals",
            "away_score",
            "ftag",
            "ag",
            "awaygoals",
        ],
    )

    if not home_team_col:
        raise ValueError(
            "Could not find a home-team column."
        )

    if not away_team_col:
        raise ValueError(
            "Could not find an away-team column."
        )

    if not home_goals_col:
        raise ValueError(
            "Could not find a home-goals column."
        )

    if not away_goals_col:
        raise ValueError(
            "Could not find an away-goals column."
        )

    rename_map = {
        home_team_col: "home_team",
        away_team_col: "away_team",
        home_goals_col: "home_goals",
        away_goals_col: "away_goals",
    }

    date_col = find_column(
        df.columns,
        [
            "date",
            "match_date",
            "datetime",
        ],
    )

    home_xg_col = find_column(
        df.columns,
        [
            "home_xg",
            "homexg",
            "home_expected_goals",
            "hxg",
        ],
    )

    away_xg_col = find_column(
        df.columns,
        [
            "away_xg",
            "awayxg",
            "away_expected_goals",
            "axg",
        ],
    )

    competition_col = find_column(
        df.columns,
        [
            "competition",
            "league",
            "tournament",
        ],
    )

    if date_col:
        rename_map[date_col] = "date"

    if home_xg_col:
        rename_map[home_xg_col] = "home_xg"

    if away_xg_col:
        rename_map[away_xg_col] = "away_xg"

    if competition_col:
        rename_map[competition_col] = "competition"

    data = df.rename(columns=rename_map).copy()

    data["home_team"] = data["home_team"].map(norm)
    data["away_team"] = data["away_team"].map(norm)

    data["home_goals"] = pd.to_numeric(
        data["home_goals"],
        errors="coerce",
    )

    data["away_goals"] = pd.to_numeric(
        data["away_goals"],
        errors="coerce",
    )

    if "date" in data.columns:
        data["date"] = pd.to_datetime(
            data["date"],
            errors="coerce",
        )

    if "home_xg" in data.columns:
        data["home_xg"] = pd.to_numeric(
            data["home_xg"],
            errors="coerce",
        )

    if "away_xg" in data.columns:
        data["away_xg"] = pd.to_numeric(
            data["away_xg"],
            errors="coerce",
        )

    data = data[
        data["home_team"].astype(bool)
        & data["away_team"].astype(bool)
        & data["home_goals"].notna()
        & data["away_goals"].notna()
    ].copy()

    data["home_goals"] = data["home_goals"].clip(0, 20)
    data["away_goals"] = data["away_goals"].clip(0, 20)

    if "date" in data.columns:
        data = data.sort_values("date")

    data = data.reset_index(drop=True)

    if len(data) < 10:
        raise ValueError(
            "Historical data needs at least 10 valid matches."
        )

    return data


def fit_strength_model(df, half_life_days=120):
    """
    Recency-weighted team strength model.

    The model estimates:
        home attack
        home defense
        away attack
        away defense

    using iterative Poisson residual updates.
    """

    data = df.copy()

    using_xg = (
        "home_xg" in data.columns
        and "away_xg" in data.columns
        and (
            data["home_xg"].notna().sum() >= 5
            and data["away_xg"].notna().sum() >= 5
        )
    )

    if using_xg:
        home_metric = data["home_xg"].fillna(data["home_goals"])
        away_metric = data["away_xg"].fillna(data["away_goals"])
    else:
        home_metric = data["home_goals"]
        away_metric = data["away_goals"]

    data = data.copy()
    data["_home_metric"] = home_metric
    data["_away_metric"] = away_metric

    # --------------------------------------------------------
    # Recency weights
    # --------------------------------------------------------

    if "date" in data.columns and data["date"].notna().any():
        latest_date = data["date"].max()

        days_old = (
            latest_date - data["date"]
        ).dt.days.fillna(0).clip(lower=0)

        weights = (
            2
            ** (-days_old / float(half_life_days))
        )
    else:
        latest_date = None

        count = len(data)

        weights = pd.Series(
            [
                0.97 ** (count - 1 - i)
                for i in range(count)
            ],
            index=data.index,
        )

    weights = weights.clip(lower=0.02)

    home_average = weighted_mean(
        data["_home_metric"].tolist(),
        weights.tolist(),
    )

    away_average = weighted_mean(
        data["_away_metric"].tolist(),
        weights.tolist(),
    )

    home_average = clamp(home_average, 0.40, 3.50)
    away_average = clamp(away_average, 0.40, 3.50)

    teams = sorted(
        set(data["home_team"]).union(
            set(data["away_team"])
        )
    )

    # Initial strengths.
    home_attack = defaultdict(lambda: 1.0)
    home_defense = defaultdict(lambda: 1.0)
    away_attack = defaultdict(lambda: 1.0)
    away_defense = defaultdict(lambda: 1.0)

    home_counts = Counter(data["home_team"])
    away_counts = Counter(data["away_team"])

    # --------------------------------------------------------
    # Iterative residual fitting
    # --------------------------------------------------------

    for _ in range(9):
        ha_sum = defaultdict(float)
        ha_weight = defaultdict(float)

        hd_sum = defaultdict(float)
        hd_weight = defaultdict(float)

        aa_sum = defaultdict(float)
        aa_weight = defaultdict(float)

        ad_sum = defaultdict(float)
        ad_weight = defaultdict(float)

        for idx, row in data.iterrows():
            h = row["home_team"]
            a = row["away_team"]
            weight = float(weights.loc[idx])

            observed_home = max(
                0.05,
                float(row["_home_metric"]),
            )

            observed_away = max(
                0.05,
                float(row["_away_metric"]),
            )

            expected_home = max(
                0.05,
                home_average
                * home_attack[h]
                * away_defense[a],
            )

            expected_away = max(
                0.05,
                away_average
                * away_attack[a]
                * home_defense[h],
            )

            home_residual = math.log(
                observed_home / expected_home
            )

            away_residual = math.log(
                observed_away / expected_away
            )

            ha_sum[h] += weight * home_residual
            ha_weight[h] += weight

            ad_sum[a] += weight * home_residual
            ad_weight[a] += weight

            aa_sum[a] += weight * away_residual
            aa_weight[a] += weight

            hd_sum[h] += weight * away_residual
            hd_weight[h] += weight

        learning_rate = 0.28

        for team in teams:
            if ha_weight[team] > 0:
                home_attack[team] *= math.exp(
                    learning_rate
                    * ha_sum[team]
                    / ha_weight[team]
                )

            if hd_weight[team] > 0:
                home_defense[team] *= math.exp(
                    learning_rate
                    * hd_sum[team]
                    / hd_weight[team]
                )

            if aa_weight[team] > 0:
                away_attack[team] *= math.exp(
                    learning_rate
                    * aa_sum[team]
                    / aa_weight[team]
                )

            if ad_weight[team] > 0:
                away_defense[team] *= math.exp(
                    learning_rate
                    * ad_sum[team]
                    / ad_weight[team]
                )

            home_attack[team] = clamp(
                home_attack[team],
                0.55,
                1.65,
            )

            home_defense[team] = clamp(
                home_defense[team],
                0.55,
                1.65,
            )

            away_attack[team] = clamp(
                away_attack[team],
                0.55,
                1.65,
            )

            away_defense[team] = clamp(
                away_defense[team],
                0.55,
                1.65,
            )

        # Normalize the four strength groups so that
        # league averages remain represented by the base rates.
        for values, counts in [
            (home_attack, home_counts),
            (home_defense, home_counts),
            (away_attack, away_counts),
            (away_defense, away_counts),
        ]:
            denominator = 0.0
            numerator = 0.0

            for team in teams:
                count = max(1, counts.get(team, 1))
                numerator += count * math.log(
                    max(0.01, values[team])
                )
                denominator += count

            geometric_mean = math.exp(
                numerator / max(1.0, denominator)
            )

            for team in teams:
                values[team] = clamp(
                    values[team] / geometric_mean,
                    0.55,
                    1.65,
                )

    # --------------------------------------------------------
    # Shrink small samples toward league average
    # --------------------------------------------------------

    team_strengths = {}

    for team in teams:
        sample_size = (
            home_counts.get(team, 0)
            + away_counts.get(team, 0)
        )

        reliability = sample_size / (
            sample_size + 10.0
        )

        def shrink(value):
            return 1.0 + reliability * (value - 1.0)

        team_strengths[team] = {
            "home_attack": shrink(home_attack[team]),
            "home_defense": shrink(home_defense[team]),
            "away_attack": shrink(away_attack[team]),
            "away_defense": shrink(away_defense[team]),
            "matches": sample_size,
        }

    return {
        "teams": team_strengths,
        "home_average": home_average,
        "away_average": away_average,
        "rows": len(data),
        "using_xg": using_xg,
        "latest_date": (
            latest_date.strftime("%Y-%m-%d")
            if latest_date is not None
            else None
        ),
    }


@st.cache_data(show_spinner=False)
def prepare_history_from_bytes(raw_bytes):
    """
    Read and prepare an uploaded historical CSV, then fit the strength model.
    """
    import io

    data = pd.read_csv(io.BytesIO(raw_bytes))

    def column_key(value):
        return "".join(
            char for char in str(value).lower()
            if char.isalnum()
        )

    def find_column(candidates):
        lookup = {
            column_key(column): column
            for column in data.columns
        }

        for candidate in candidates:
            key = column_key(candidate)
            if key in lookup:
                return lookup[key]

        return None

    home_team_col = find_column([
        "home_team",
        "hometeam",
        "home",
        "home_team_name"
    ])

    away_team_col = find_column([
        "away_team",
        "awayteam",
        "away",
        "away_team_name"
    ])

    home_goals_col = find_column([
        "home_goals",
        "homegoals",
        "fthg",
        "home_score",
        "homescore",
        "hg"
    ])

    away_goals_col = find_column([
        "away_goals",
        "awaygoals",
        "ftag",
        "away_score",
        "awayscore",
        "ag"
    ])

    if not home_team_col:
        raise ValueError(
            "Could not find the home-team column. "
            "Use a column such as home_team or HomeTeam."
        )

    if not away_team_col:
        raise ValueError(
            "Could not find the away-team column. "
            "Use a column such as away_team or AwayTeam."
        )

    if not home_goals_col:
        raise ValueError(
            "Could not find the home-goals column. "
            "Use home_goals or FTHG."
        )

    if not away_goals_col:
        raise ValueError(
            "Could not find the away-goals column. "
            "Use away_goals or FTAG."
        )

    rename_map = {
        home_team_col: "home_team",
        away_team_col: "away_team",
        home_goals_col: "home_goals",
        away_goals_col: "away_goals",
    }

    date_col = find_column([
        "date",
        "match_date",
        "datetime"
    ])

    home_xg_col = find_column([
        "home_xg",
        "homexg",
        "hxg",
        "home_expected_goals",
        "expected_home_goals"
    ])

    away_xg_col = find_column([
        "away_xg",
        "awayxg",
        "axg",
        "away_expected_goals",
        "expected_away_goals"
    ])

    competition_col = find_column([
        "competition",
        "league",
        "tournament"
    ])

    if date_col:
        rename_map[date_col] = "date"

    if home_xg_col:
        rename_map[home_xg_col] = "home_xg"

    if away_xg_col:
        rename_map[away_xg_col] = "away_xg"

    if competition_col:
        rename_map[competition_col] = "competition"

    data = data.rename(columns=rename_map)

    data["home_team"] = (
        data["home_team"]
        .astype(str)
        .str.strip()
        .map(norm)
    )

    data["away_team"] = (
        data["away_team"]
        .astype(str)
        .str.strip()
        .map(norm)
    )

    data["home_goals"] = pd.to_numeric(
        data["home_goals"],
        errors="coerce"
    )

    data["away_goals"] = pd.to_numeric(
        data["away_goals"],
        errors="coerce"
    )

    data = data.dropna(
        subset=[
            "home_team",
            "away_team",
            "home_goals",
            "away_goals"
        ]
    )

    data = data[
        (data["home_goals"] >= 0)
        & (data["away_goals"] >= 0)
    ].copy()

    if "date" in data.columns:
        data["date"] = pd.to_datetime(
            data["date"],
            errors="coerce"
        )

    if "home_xg" in data.columns:
        data["home_xg"] = pd.to_numeric(
            data["home_xg"],
            errors="coerce"
        )

    if "away_xg" in data.columns:
        data["away_xg"] = pd.to_numeric(
            data["away_xg"],
            errors="coerce"
        )

    if "competition" in data.columns:
        data["competition"] = (
            data["competition"]
            .astype(str)
            .str.strip()
        )

    data = data.reset_index(drop=True)

    if len(data) < 10:
        raise ValueError(
            "The CSV contains fewer than 10 valid matches. "
            "Upload a larger historical dataset."
        )

    model = fit_strength_model(data)

    return model


def clamp(value, low, high):
    return max(low, min(high, value))


def form_adjustment(entries):
    """
    Convert recent results into modest attack/defence multipliers.

    Most recent result gets the highest weight.
    """
    if not entries:
        return 1.0, 1.0, None

    n = len(entries)

    weights = []
    for index in range(n):
        age = n - 1 - index
        weights.append(0.72 ** age)

    total_weight = sum(weights)

    gf_avg = sum(
        entry[0] * weights[index]
        for index, entry in enumerate(entries)
    ) / total_weight

    ga_avg = sum(
        entry[1] * weights[index]
        for index, entry in enumerate(entries)
    ) / total_weight

    points = []

    for gf, ga in entries:
        if gf > ga:
            points.append(3)
        elif gf == ga:
            points.append(1)
        else:
            points.append(0)

    points_avg = sum(
        points[index] * weights[index]
        for index in range(n)
    ) / total_weight

    attack_multiplier = clamp(
        1.0
        + (gf_avg - 1.35) * 0.06
        + (points_avg - 1.35) * 0.02,
        0.90,
        1.10
    )

    defence_multiplier = clamp(
        1.0
        + (ga_avg - 1.35) * 0.05,
        0.92,
        1.08
    )

    summary = {
        "matches": n,
        "gf_avg": gf_avg,
        "ga_avg": ga_avg,
        "points_avg": points_avg,
    }

    return (
        attack_multiplier,
        defence_multiplier,
        summary
    )


FORMATION_EFFECTS = {
    "4-3-3": (1.015, 1.005),
    "4-2-3-1": (1.010, 1.010),
    "4-4-2": (1.000, 1.000),
    "3-5-2": (1.005, 1.015),
    "3-4-3": (1.015, 1.000),
    "5-3-2": (0.985, 1.025),
    "5-4-1": (0.970, 1.035),
    "4-5-1": (0.985, 1.025),
    "Other / Unknown": (1.000, 1.000),
}


MOTIVATION_EFFECTS = {
    "Very Low": 0.975,
    "Low": 0.988,
    "Normal": 1.000,
    "High": 1.012,
    "Very High": 1.025,
}


ABSENCE_EFFECTS = {
    "Not specified": (1.000, 1.000),
    "Attackers": (0.970, 1.000),
    "Midfielders": (0.985, 1.010),
    "Defenders": (1.000, 1.030),
    "Goalkeeper": (1.000, 1.040),
    "Mixed": (0.975, 1.025),
}


def team_context_multiplier(
    formation,
    motivation,
    absence_count,
    absence_importance,
    absence_type,
    form_entries,
    form_weight,
    rest_days,
    match_stage,
    aggregate_for,
    aggregate_against,
):
    """
    Return attack and defence multipliers for one team.
    """

    attack_multiplier = 1.0
    defence_multiplier = 1.0

    formation_attack, formation_defence = FORMATION_EFFECTS.get(
        formation,
        (1.0, 1.0)
    )

    attack_multiplier *= formation_attack
    defence_multiplier *= formation_defence

    attack_multiplier *= MOTIVATION_EFFECTS.get(
        motivation,
        1.0
    )

    absence_attack, absence_defence = ABSENCE_EFFECTS.get(
        absence_type,
        (1.0, 1.0)
    )

    importance = clamp(
        absence_importance / 100.0,
        0.0,
        1.0
    )

    count_factor = clamp(
        absence_count,
        0,
        5
    )

    absence_scale = min(
        1.0,
        count_factor / 3.0
    ) * importance

    attack_multiplier *= (
        1.0
        + (absence_attack - 1.0) * absence_scale
    )

    defence_multiplier *= (
        1.0
        + (absence_defence - 1.0) * absence_scale
    )

    if form_entries:
        form_attack, form_defence, _ = form_adjustment(
            form_entries
        )

        form_weight_factor = clamp(
            form_weight / 100.0,
            0.0,
            1.0
        )

        attack_multiplier *= (
            1.0
            + (form_attack - 1.0)
            * form_weight_factor
        )

        defence_multiplier *= (
            1.0
            + (form_defence - 1.0)
            * form_weight_factor
        )

    # Rest/fatigue adjustment.
    # Five days is treated as neutral.
    rest_delta = clamp(
        (rest_days - 5) * 0.005,
        -0.025,
        0.015
    )

    attack_multiplier *= 1.0 + rest_delta

    # Fatigue tends to hurt defensive concentration slightly more.
    defence_multiplier *= 1.0 - rest_delta * 0.65

    # Knockout second-leg aggregate pressure.
    if match_stage == "Knockout — second leg":
        if aggregate_for < aggregate_against:
            attack_multiplier *= 1.045
            defence_multiplier *= 1.015

        elif aggregate_for > aggregate_against:
            attack_multiplier *= 0.985
            defence_multiplier *= 0.990

        else:
            attack_multiplier *= 1.015

    attack_multiplier = clamp(
        attack_multiplier,
        0.80,
        1.20
    )

    defence_multiplier = clamp(
        defence_multiplier,
        0.85,
        1.20
    )

    return (
        attack_multiplier,
        defence_multiplier
    )


def conditions_multiplier(weather, pitch):
    """
    Small environmental adjustments.
    These are intentionally conservative.
    """

    weather_effects = {
        "Clear": 1.000,
        "Cloudy": 1.000,
        "Light Rain": 0.985,
        "Heavy Rain": 0.950,
        "Strong Wind": 0.940,
        "Extreme Heat": 0.965,
        "Snow": 0.930,
    }

    pitch_effects = {
        "Excellent": 1.005,
        "Good": 1.000,
        "Average": 0.990,
        "Poor": 0.960,
    }

    weather_factor = weather_effects.get(
        weather,
        1.000
    )

    pitch_factor = pitch_effects.get(
        pitch,
        1.000
    )

    return weather_factor * pitch_factor


def get_team_strength(model, team_key):
    """
    Return strengths from the historical model.
    Unknown teams are treated as average rather than inventing
    historical strength.
    """

    if model is not None:
        return model["teams"].get(
            team_key,
            {
                "home_attack": 1.0,
                "home_defense": 1.0,
                "away_attack": 1.0,
                "away_defense": 1.0,
                "matches": 0,
            }
        )

    data = TEAM_DATA.get(team_key)

    if data is None:
        return {
            "home_attack": 1.0,
            "home_defense": 1.0,
            "away_attack": 1.0,
            "away_defense": 1.0,
            "matches": 0,
        }

    # Convert the built-in illustrative ratings into relative strengths.
    return {
        "home_attack": data["attack"] / 1.50,
        "home_defense": data["defense"] / 1.25,
        "away_attack": data["attack"] / 1.50,
        "away_defense": data["defense"] / 1.25,
        "matches": 0,
    }


def predict_expected_goals(
    home_key,
    away_key,
    venue,
    model,
    home_attack_context,
    home_defence_context,
    away_attack_context,
    away_defence_context,
    weather,
    pitch,
):
    """
    Produce expected goals using either:
    1. historical CSV strength model, or
    2. built-in fallback ratings.
    """

    home_strength = get_team_strength(
        model,
        home_key
    )

    away_strength = get_team_strength(
        model,
        away_key
    )

    if model is not None:
        home_average = model["home_average"]
        away_average = model["away_average"]

        if venue == "Home team venue":
            base_home = home_average
            base_away = away_average

            home_attack = home_strength["home_attack"]
            home_defence = home_strength["home_defense"]

            away_attack = away_strength["away_attack"]
            away_defence = away_strength["away_defense"]

        elif venue == "Away team venue":
            base_home = away_average
            base_away = home_average

            home_attack = home_strength["away_attack"]
            home_defence = home_strength["away_defense"]

            away_attack = away_strength["home_attack"]
            away_defence = away_strength["home_defense"]

        elif venue == "Neutral venue":
            neutral_average = (
                home_average + away_average
            ) / 2.0

            base_home = neutral_average
            base_away = neutral_average

            home_attack = (
                home_strength["home_attack"]
                + home_strength["away_attack"]
            ) / 2.0

            home_defence = (
                home_strength["home_defense"]
                + home_strength["away_defense"]
            ) / 2.0

            away_attack = (
                away_strength["home_attack"]
                + away_strength["away_attack"]
            ) / 2.0

            away_defence = (
                away_strength["home_defense"]
                + away_strength["away_defense"]
            ) / 2.0

        else:
            base_home = home_average
            base_away = away_average

            home_attack = home_strength["home_attack"]
            home_defence = home_strength["home_defense"]

            away_attack = away_strength["away_attack"]
            away_defence = away_strength["away_defense"]

        home_xg = (
            base_home
            * home_attack
            * away_defence
            * home_attack_context
            * away_defence_context
        )

        away_xg = (
            base_away
            * away_attack
            * home_defence
            * away_attack_context
            * home_defence_context
        )

    else:
        # Built-in illustrative model.
        fallback_home_average = 1.50
        fallback_away_average = 1.20

        if venue == "Home team venue":
            base_home = fallback_home_average
            base_away = fallback_away_average

            home_attack = home_strength["home_attack"]
            home_defence = home_strength["home_defense"]

            away_attack = away_strength["away_attack"]
            away_defence = away_strength["away_defense"]

        elif venue == "Away team venue":
            base_home = fallback_away_average
            base_away = fallback_home_average

            home_attack = home_strength["away_attack"]
            home_defence = home_strength["away_defense"]

            away_attack = away_strength["home_attack"]
            away_defence = away_strength["home_defense"]

        elif venue == "Neutral venue":
            neutral_average = (
                fallback_home_average
                + fallback_away_average
            ) / 2.0

            base_home = neutral_average
            base_away = neutral_average

            home_attack = (
                home_strength["home_attack"]
                + home_strength["away_attack"]
            ) / 2.0

            home_defence = (
                home_strength["home_defense"]
                + home_strength["away_defense"]
            ) / 2.0

            away_attack = (
                away_strength["home_attack"]
                + away_strength["away_attack"]
            ) / 2.0

            away_defence = (
                away_strength["home_defense"]
                + away_strength["away_defense"]
            ) / 2.0

        else:
            base_home = fallback_home_average
            base_away = fallback_away_average

            home_attack = home_strength["home_attack"]
            home_defence = home_strength["home_defense"]

            away_attack = away_strength["away_attack"]
            away_defence = away_strength["away_defense"]

        home_xg = (
            base_home
            * home_attack
            * away_defence
            * home_attack_context
            * away_defence_context
        )

        away_xg = (
            base_away
            * away_attack
            * home_defence
            * away_attack_context
            * home_defence_context
        )

        # Preserve the built-in home advantage only when the
        # home team is actually at home.
        if venue == "Home team venue":
            home_data = TEAM_DATA.get(home_key)

            if home_data:
                home_xg += home_data["home_adv"]

    condition_factor = conditions_multiplier(
        weather,
        pitch
    )

    home_xg *= condition_factor
    away_xg *= condition_factor

    home_xg = clamp(
        home_xg,
        0.05,
        5.00
    )

    away_xg = clamp(
        away_xg,
        0.05,
        5.00
    )

    return home_xg, away_xg


def calculate_matrix(
    home_xg,
    away_xg,
    max_goals=8,
    rho=-0.05
):
    """
    Dixon-Coles-style score probability matrix.
    """

    matrix = []

    for home_goals in range(max_goals + 1):
        row = []

        for away_goals in range(max_goals + 1):
            probability = (
                poisson_pmf(home_goals, home_xg)
                * poisson_pmf(away_goals, away_xg)
            )

            tau = 1.0

            if home_goals == 0 and away_goals == 0:
                tau = 1.0 - (
                    home_xg
                    * away_xg
                    * rho
                )

            elif home_goals == 0 and away_goals == 1:
                tau = 1.0 + away_xg * rho

            elif home_goals == 1 and away_goals == 0:
                tau = 1.0 + home_xg * rho

            elif home_goals == 1 and away_goals == 1:
                tau = 1.0 - rho

            row.append(
                max(
                    0.0,
                    probability * tau
                )
            )

        matrix.append(row)

    total_probability = sum(
        sum(row)
        for row in matrix
    )

    if total_probability <= 0:
        total_probability = 1.0

    matrix = [
        [
            value / total_probability
            for value in row
        ]
        for row in matrix
    ]

    home_win = 0.0
    draw = 0.0
    away_win = 0.0

    over_05 = 0.0
    over_15 = 0.0
    over_25 = 0.0
    over_35 = 0.0
    over_45 = 0.0

    btts_yes = 0.0
    home_clean_sheet = 0.0
    away_clean_sheet = 0.0

    scorelines = []

    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            probability = matrix[
                home_goals
            ][
                away_goals
            ]

            total_goals = (
                home_goals
                + away_goals
            )

            if home_goals > away_goals:
                home_win += probability

            elif home_goals == away_goals:
                draw += probability

            else:
                away_win += probability

            if total_goals >= 1:
                over_05 += probability

            if total_goals >= 2:
                over_15 += probability

            if total_goals >= 3:
                over_25 += probability

            if total_goals >= 4:
                over_35 += probability

            if total_goals >= 5:
                over_45 += probability

            if (
                home_goals >= 1
                and away_goals >= 1
            ):
                btts_yes += probability

            if away_goals == 0:
                home_clean_sheet += probability

            if home_goals == 0:
                away_clean_sheet += probability

            scorelines.append(
                (
                    home_goals,
                    away_goals,
                    probability
                )
            )

    scorelines.sort(
        key=lambda item: item[2],
        reverse=True
    )

    return {
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,

        "over_05": over_05,
        "over_15": over_15,
        "over_25": over_25,
        "over_35": over_35,
        "over_45": over_45,

        "under_05": 1.0 - over_05,
        "under_15": 1.0 - over_15,
             "under_25": 1.0 - over_25,
        "under_35": 1.0 - over_35,
        "under_45": 1.0 - over_45,

        "btts_yes": btts_yes,
        "btts_no": 1.0 - btts_yes,

        "home_clean_sheet": home_clean_sheet,
        "away_clean_sheet": away_clean_sheet,

        "home_or_draw": home_win + draw,
        "draw_or_away": draw + away_win,
        "home_or_away": home_win + away_win,

        "top_scores": scorelines[:10],
        "matrix": matrix,
    }


def poisson_sample(lam, rng):
    """
    Knuth Poisson sampler.
    """

    if lam <= 0:
        return 0

    limit = math.exp(-lam)

    product = 1.0
    count = 0

    while product > limit:
        count += 1
        product *= rng.random()

    return count - 1


def run_monte_carlo(
    home_xg,
    away_xg,
    n_sims,
    seed
):
    rng = random.Random(seed)

    home_wins = 0
    draws = 0
    away_wins = 0

    over_05 = 0
    over_15 = 0
    over_25 = 0
    over_35 = 0

    btts = 0

    score_counter = Counter()

    total_goals = 0

    for _ in range(n_sims):

        home_goals = poisson_sample(
            home_xg,
            rng
        )

        away_goals = poisson_sample(
            away_xg,
            rng
        )

        score_counter[
            (home_goals, away_goals)
        ] += 1

        goals = (
            home_goals
            + away_goals
        )

        total_goals += goals

        if home_goals > away_goals:
            home_wins += 1

        elif home_goals == away_goals:
            draws += 1

        else:
            away_wins += 1

        if goals >= 1:
            over_05 += 1

        if goals >= 2:
            over_15 += 1

        if goals >= 3:
            over_25 += 1

        if goals >= 4:
            over_35 += 1

        if (
            home_goals > 0
            and away_goals > 0
        ):
            btts += 1

    def probability(count):
        return count / n_sims

    return {
        "home_win": probability(home_wins),
        "draw": probability(draws),
        "away_win": probability(away_wins),

        "over_05": probability(over_05),
        "over_15": probability(over_15),
        "over_25": probability(over_25),
        "over_35": probability(over_35),

        "btts_yes": probability(btts),

        "average_goals": (
            total_goals / n_sims
        ),

        "top_scores": [
            (
                home_goals,
                away_goals,
                count / n_sims
            )
            for (
                home_goals,
                away_goals
            ), count in score_counter.most_common(10)
        ],

        "samples": n_sims,
    }


def mc_standard_error(
    probability,
    samples
):
    if samples <= 0:
        return 0.0

    return math.sqrt(
        probability
        * (1.0 - probability)
        / samples
    )


def data_coverage_score(
    history_model,
    home_key,
    away_key,
    home_form,
    away_form,
    home_lineup,
    away_lineup,
    venue,
    weather,
    pitch,
):
    """
    Data coverage is NOT a guarantee of prediction accuracy.
    """

    score = 0

    if history_model is not None:

        rows = history_model["rows"]

        if rows >= 1000:
            score += 45

        elif rows >= 500:
            score += 40

        elif rows >= 200:
            score += 35

        elif rows >= 50:
            score += 27

        else:
            score += 18

        home_sample = (
            history_model["teams"]
            .get(home_key, {})
            .get("matches", 0)
        )

        away_sample = (
            history_model["teams"]
            .get(away_key, {})
            .get("matches", 0)
        )

        if home_sample >= 10:
            score += 8

        elif home_sample >= 5:
            score += 4

        if away_sample >= 10:
            score += 8

        elif away_sample >= 5:
            score += 4

    else:

        if home_key in TEAM_DATA:
            score += 12

        if away_key in TEAM_DATA:
            score += 12

    if home_form:
        score += 6

    if away_form:
        score += 6

    if home_lineup.strip():
        score += 3

    if away_lineup.strip():
        score += 3

    if venue != "Not sure":
        score += 3

    if weather != "Clear":
        score += 1

    if pitch != "Good":
        score += 1

    score = int(
        clamp(
            score,
            0,
            100
        )
    )

    if score >= 75:
        level = "High"

    elif score >= 50:
        level = "Moderate"

    else:
        level = "Low"

    return score, level


def format_percent(value):
    return f"{value * 100:.1f}%"


# ============================================================
# APP UI
# ============================================================

st.markdown(
    """
    <div class="title">MATCHLAB Pro</div>
    <div class="subtitle">
        Context-aware football probability and simulation workspace
        · 2026/2027
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="welcome">
        <b>Welcome to MATCHLAB Pro V2.</b><br>
        Build a match using team strength, recent form,
        tactical setup, absences, venue, rest, weather
        and optional historical data.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("MATCH SETUP")

    simulation_mode = st.selectbox(
        "Simulation mode",
        [
            "Standard",
            "Competition-Aware",
            "Lineup-Based",
            "Full Context",
            "Custom Simulation",
        ],
        index=3
    )

    team_options = [
        "Aston Villa",
        "Manchester City",
        "Arsenal",
        "Liverpool",
        "Chelsea",
        "Manchester United",
        "Real Madrid",
        "Barcelona",
        "Bayern Munich",
        "Inter Milan",
        "Juventus",
        "PSG",
    ]

    home_team = st.selectbox(
        "Home team",
        team_options,
        index=0
    )

    away_team = st.selectbox(
        "Away team",
        team_options,
        index=1
    )

    st.divider()

    competition = st.selectbox(
        "Competition",
        COMPETITIONS
    )

    season_choice = st.selectbox(
        "Season",
        [
            "2026/2027",
            "2025/2026",
            "Other / Custom",
        ]
    )

    custom_season = ""

    if season_choice == "Other / Custom":

        custom_season = st.text_input(
            "Custom season",
            placeholder="e.g. 2024/2025"
        )

    match_stage = st.selectbox(
        "Match stage",
        [
            "Regular season",
            "Group stage",
            "Knockout — first leg",
            "Knockout — second leg",
            "Quarter-final",
            "Semi-final",
            "Final",
            "Friendly",
        ]
    )

    venue = st.selectbox(
        "Venue",
        [
            "Home team venue",
            "Away team venue",
            "Neutral venue",
            "Not sure",
        ]
    )

    max_goals = st.slider(
        "Maximum goals in score matrix",
        min_value=5,
        max_value=12,
        value=8
    )

    form_weight = st.slider(
        "Global recent-form weight",
        min_value=0,
        max_value=100,
        value=70,
        step=5
    )

    simulation_count = st.selectbox(
        "Monte Carlo simulations",
        [
            1000,
            5000,
            10000,
            25000,
            50000,
        ],
        index=2
    )

    random_seed = st.number_input(
        "Random seed",
        min_value=1,
        max_value=999999,
        value=42
    )

    st.divider()

    st.header("HISTORICAL DATA")

    history_file = st.file_uploader(
        "Optional historical CSV",
        type=["csv"],
        help=(
            "Required columns: home_team, away_team, "
            "home_goals, away_goals. "
            "Optional: date, home_xg, away_xg."
        )
    )

    st.caption(
        "Upload historical matches to replace the built-in "
        "illustrative team ratings with a fitted strength model."
    )


# ============================================================
# HISTORICAL MODEL
# ============================================================

history_model = None
history_error = None

if history_file is not None:

    try:

        history_model = prepare_history_from_bytes(
            history_file.getvalue()
        )

    except Exception as error:

        history_error = str(error)

        st.sidebar.error(
            f"Historical data error: {history_error}"
        )


if history_model is not None:

    st.sidebar.success(
        f"Historical model active · "
        f"{history_model['rows']} matches"
    )

    if history_model["using_xg"]:

        st.sidebar.caption(
            "Model metric: historical xG"
        )

    else:

        st.sidebar.caption(
            "Model metric: historical goals"
        )

    if history_model["latest_date"]:

        st.sidebar.caption(
            f"Latest match: "
            f"{history_model['latest_date']}"
        )


# ============================================================
# VALIDATION
# ============================================================

home_key = norm(home_team)
away_key = norm(away_team)

if home_key == away_key:

    st.error(
        "Home and away teams must be different."
    )

    st.stop()


# ============================================================
# PERFORMANCE EVIDENCE
# ============================================================

st.subheader("1. Performance Evidence")

col1, col2 = st.columns(2)

with col1:

    st.markdown("### Home team form")

    home_form_text = st.text_area(
        "Recent results",
        placeholder="2-1, 1-1, 3-0",
        height=100,
        key="home_form"
    )

    home_form_weight = st.slider(
        "Home form influence",
        min_value=0,
        max_value=100,
        value=70,
        step=5,
        key="home_form_weight"
    )

with col2:

    st.markdown("### Away team form")

    away_form_text = st.text_area(
        "Recent results",
        placeholder="1-0, 2-2, 0-1",
        height=100,
        key="away_form"
    )

    away_form_weight = st.slider(
        "Away form influence",
        min_value=0,
        max_value=100,
        value=70,
        step=5,
        key="away_form_weight"
    )


# ============================================================
# LINEUPS
# ============================================================

st.subheader("2. Lineup & Availability")

lineup_col1, lineup_col2 = st.columns(2)

with lineup_col1:

    st.markdown("### Home lineup")

    home_lineup_image = st.file_uploader(
        "Upload lineup image",
        type=[
            "png",
            "jpg",
            "jpeg",
            "webp"
        ],
        key="home_lineup_image"
    )

    if home_lineup_image is not None:

        st.image(
            home_lineup_image,
            caption=f"{home_team} lineup",
            use_container_width=True
        )

    home_lineup_notes = st.text_area(
        "Lineup notes",
        placeholder=(
            "Example: first-choice striker starts; "
            "two defenders rotated."
        ),
        height=110,
        key="home_lineup_notes"
    )


with lineup_col2:

    st.markdown("### Away lineup")

    away_lineup_image = st.file_uploader(
        "Upload lineup image",
        type=[
            "png",
            "jpg",
            "jpeg",
            "webp"
        ],
        key="away_lineup_image"
    )

    if away_lineup_image is not None:

        st.image(
            away_lineup_image,
            caption=f"{away_team} lineup",
            use_container_width=True
        )

    away_lineup_notes = st.text_area(
        "Lineup notes",
        placeholder=(
            "Example: starting goalkeeper absent; "
            "midfield unchanged."
        ),
        height=110,
        key="away_lineup_notes"
    )


st.info(
    "Lineup images are displayed for reference. "
    "MATCHLAB does not automatically read names from the image. "
    "Use the structured absence fields below to influence the model."
)


# ============================================================
# MATCH CONTEXT
# ============================================================

st.subheader("3. Match Context")

context_home, context_away = st.columns(2)

with context_home:

    st.markdown(f"### {home_team}")

    home_formation = st.selectbox(
        "Formation",
        FORMATIONS,
        key="home_formation"
    )

    home_motivation = st.selectbox(
        "Motivation",
        [
            "Very Low",
            "Low",
            "Normal",
            "High",
            "Very High",
        ],
        index=2,
        key="home_motivation"
    )

    home_absences = st.number_input(
        "Unavailable players",
        min_value=0,
        max_value=8,
        value=0,
        key="home_absences"
    )

    home_absence_type = st.selectbox(
        "Main absence profile",
        [
            "Not specified",
            "Attackers",
            "Midfielders",
            "Defenders",
            "Goalkeeper",
            "Mixed",
        ],
        key="home_absence_type"
    )

    home_absence_importance = st.slider(
        "Average absence importance",
        0,
        100,
        50,
        5,
        key="home_absence_importance"
    )

    home_absence_notes = st.text_input(
        "Absence notes",
        placeholder="Optional",
        key="home_absence_notes"
    )

    home_rest_days = st.number_input(
        "Rest days",
        min_value=0,
        max_value=14,
        value=5,
        key="home_rest_days"
    )


with context_away:

    st.markdown(f"### {away_team}")

    away_formation = st.selectbox(
        "Formation",
        FORMATIONS,
        key="away_formation"
    )

    away_motivation = st.selectbox(
        "Motivation",
        [
            "Very Low",
            "Low",
            "Normal",
            "High",
            "Very High",
        ],
        index=2,
        key="away_motivation"
    )

    away_absences = st.number_input(
        "Unavailable players",
        min_value=0,
        max_value=8,
        value=0,
        key="away_absences"
    )

    away_absence_type = st.selectbox(
        "Main absence profile",
        [
            "Not specified",
            "Attackers",
            "Midfielders",
            "Defenders",
            "Goalkeeper",
            "Mixed",
        ],
        key="away_absence_type"
    )

    away_absence_importance = st.slider(
        "Average absence importance",
        0,
        100,
        50,
        5,
        key="away_absence_importance"
    )

    away_absence_notes = st.text_input(
        "Absence notes",
        placeholder="Optional",
        key="away_absence_notes"
    )

    away_rest_days = st.number_input(
        "Rest days",
        min_value=0,
        max_value=14,
        value=5,
        key="away_rest_days"
    )


# ============================================================
# SECOND LEG AGGREGATE
# ============================================================

aggregate_home = 0
aggregate_away = 0

if match_stage == "Knockout — second leg":

    st.markdown("### Aggregate score")

    aggregate_col1, aggregate_col2 = (
        st.columns(2)
    )

    with aggregate_col1:

        aggregate_home = st.number_input(
            f"{home_team} aggregate goals",
            min_value=0,
            max_value=20,
            value=0,
            key="aggregate_home"
        )

    with aggregate_col2:

        aggregate_away = st.number_input(
            f"{away_team} aggregate goals",
            min_value=0,
            max_value=20,
            value=0,
            key="aggregate_away"
        )


# ============================================================
# WEATHER / PITCH
# ============================================================

st.markdown("### Environment")

environment_col1, environment_col2 = (
    st.columns(2)
)

with environment_col1:

    weather = st.selectbox(
        "Weather",
        [
            "Clear",
            "Cloudy",
            "Light Rain",
            "Heavy Rain",
            "Strong Wind",
            "Extreme Heat",
            "Snow",
        ]
    )

with environment_col2:

    pitch = st.selectbox(
        "Pitch condition",
        [
            "Excellent",
            "Good",
            "Average",
            "Poor",
        ],
        index=1
    )


# ============================================================
# RUN BUTTON
# ============================================================

st.divider()

run_prediction = st.button(
    "RUN MATCHLAB V2",
    type="primary",
    use_container_width=True
)


if run_prediction:

    # --------------------------------------------------------
    # Parse form
    # --------------------------------------------------------

    try:

        parsed_home_form = parse_form(
            home_form_text
        )

        parsed_away_form = parse_form(
            away_form_text
        )

    except ValueError as error:

        st.error(
            f"Form input error: {error}"
        )

        st.stop()


    # --------------------------------------------------------
    # Context multipliers
    # --------------------------------------------------------

    (
        home_attack_context,
        home_defence_context
    ) = team_context_multiplier(
        formation=home_formation,
        motivation=home_motivation,
        absence_count=home_absences,
        absence_importance=home_absence_importance,
        absence_type=home_absence_type,
        form_entries=parsed_home_form,
        form_weight=(
            form_weight
            * home_form_weight
            / 100.0
        ),
        rest_days=home_rest_days,
        match_stage=match_stage,
        aggregate_for=aggregate_home,
        aggregate_against=aggregate_away,
    )


    (
        away_attack_context,
        away_defence_context
    ) = team_context_multiplier(
        formation=away_formation,
        motivation=away_motivation,
        absence_count=away_absences,
        absence_importance=away_absence_importance,
        absence_type=away_absence_type,
        form_entries=parsed_away_form,
        form_weight=(
            form_weight
            * away_form_weight
            / 100.0
        ),
        rest_days=away_rest_days,
        match_stage=match_stage,
        aggregate_for=aggregate_away,
        aggregate_against=aggregate_home,
    )


    # --------------------------------------------------------
    # Expected goals
    # --------------------------------------------------------

    home_xg, away_xg = predict_expected_goals(
        home_key=home_key,
        away_key=away_key,
        venue=venue,
        model=history_model,
        home_attack_context=home_attack_context,
        home_defence_context=home_defence_context,
        away_attack_context=away_attack_context,
        away_defence_context=away_defence_context,
        weather=weather,
        pitch=pitch,
    )


    # --------------------------------------------------------
    # Exact probability matrix
    # --------------------------------------------------------

    matrix_result = calculate_matrix(
        home_xg=home_xg,
        away_xg=away_xg,
        max_goals=max_goals,
        rho=-0.05
    )


    # --------------------------------------------------------
    # Monte Carlo
    # --------------------------------------------------------

    simulation_result = run_monte_carlo(
        home_xg=home_xg,
        away_xg=away_xg,
        n_sims=int(simulation_count),
        seed=int(random_seed)
    )


    # ---------------
        # --------------------------------------------------------
    # Data coverage
    # --------------------------------------------------------

    coverage_score, coverage_level = data_coverage_score(
        history_model=history_model,
        home_key=home_key,
        away_key=away_key,
        home_form=parsed_home_form,
        away_form=parsed_away_form,
        home_lineup=home_lineup_notes,
        away_lineup=away_lineup_notes,
        venue=venue,
        weather=weather,
        pitch=pitch,
    )

    # --------------------------------------------------------
    # Main result
    # --------------------------------------------------------

    st.success("MATCHLAB V2 prediction completed.")

    if history_model is not None:
        st.info(
            "Historical-data model active. "
            "Team strengths are estimated from the uploaded "
            "dataset using recency weighting and sample-size shrinkage."
        )
    else:
        st.warning(
            "No historical CSV was uploaded. "
            "The prediction is using MATCHLAB's built-in illustrative ratings."
        )

    # --------------------------------------------------------
    # Expected goals
    # --------------------------------------------------------

    st.subheader("4. Expected Goals")

    xg1, xg2, xg3 = st.columns(3)

    with xg1:
        st.metric(
            home_team,
            f"{home_xg:.2f} xG"
        )

    with xg2:
        st.metric(
            "Total xG",
            f"{home_xg + away_xg:.2f}"
        )

    with xg3:
        st.metric(
            away_team,
            f"{away_xg:.2f} xG"
        )

    # --------------------------------------------------------
    # 1X2 probabilities
    # --------------------------------------------------------

    st.subheader("5. Match Result Probability")

    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric(
            f"{home_team} Win",
            format_percent(
                matrix_result["home_win"]
            )
        )

    with r2:
        st.metric(
            "Draw",
            format_percent(
                matrix_result["draw"]
            )
        )

    with r3:
        st.metric(
            f"{away_team} Win",
            format_percent(
                matrix_result["away_win"]
            )
        )

    # --------------------------------------------------------
    # Monte Carlo
    # --------------------------------------------------------

    st.markdown("### Monte Carlo Simulation")

    mc1, mc2, mc3 = st.columns(3)

    with mc1:
        st.metric(
            "MC Home Win",
            format_percent(
                simulation_result["home_win"]
            )
        )

    with mc2:
        st.metric(
            "MC Draw",
            format_percent(
                simulation_result["draw"]
            )
        )

    with mc3:
        st.metric(
            "MC Away Win",
            format_percent(
                simulation_result["away_win"]
            )
        )

    agreement = max(
        abs(
            matrix_result[key]
            - simulation_result[key]
        )
        for key in [
            "home_win",
            "draw",
            "away_win"
        ]
    )

    st.caption(
        f"Exact vs Monte Carlo maximum difference: "
        f"{agreement * 100:.2f} percentage points."
    )

    # --------------------------------------------------------
    # Goal markets
    # --------------------------------------------------------

    st.subheader("6. Goal Markets")

    g1, g2, g3, g4 = st.columns(4)

    with g1:
        st.metric(
            "Over 1.5",
            format_percent(
                matrix_result["over_15"]
            )
        )

    with g2:
        st.metric(
            "Over 2.5",
            format_percent(
                matrix_result["over_25"]
            )
        )

    with g3:
        st.metric(
            "Over 3.5",
            format_percent(
                matrix_result["over_35"]
            )
        )

    with g4:
        st.metric(
            "BTTS",
            format_percent(
                matrix_result["btts_yes"]
            )
        )

    market_rows = [
        ["Over 0.5", matrix_result["over_05"]],
        ["Under 0.5", matrix_result["under_05"]],
        ["Over 1.5", matrix_result["over_15"]],
        ["Under 1.5", matrix_result["under_15"]],
        ["Over 2.5", matrix_result["over_25"]],
        ["Under 2.5", matrix_result["under_25"]],
        ["Over 3.5", matrix_result["over_35"]],
        ["Under 3.5", matrix_result["under_35"]],
        ["Over 4.5", matrix_result["over_45"]],
        ["Under 4.5", matrix_result["under_45"]],
        ["BTTS Yes", matrix_result["btts_yes"]],
        ["BTTS No", matrix_result["btts_no"]],
        [
            f"{home_team} Clean Sheet",
            matrix_result["home_clean_sheet"]
        ],
        [
            f"{away_team} Clean Sheet",
            matrix_result["away_clean_sheet"]
        ],
        [
            "Home or Draw",
            matrix_result["home_or_draw"]
        ],
        [
            "Draw or Away",
            matrix_result["draw_or_away"]
        ],
        [
            "Home or Away",
            matrix_result["home_or_away"]
        ],
    ]

    market_df = pd.DataFrame(
        market_rows,
        columns=[
            "Market",
            "Probability"
        ]
    )

    market_df["Probability"] = market_df[
        "Probability"
    ].apply(
        lambda x: f"{x * 100:.1f}%"
    )

    st.dataframe(
        market_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # Correct scores
    # --------------------------------------------------------

    st.subheader("7. Most Likely Correct Scores")

    score_rows = []

    for (
        hg,
        ag,
        probability
    ) in matrix_result["top_scores"][:10]:

        score_rows.append(
            {
                "Score": f"{hg}-{ag}",
                "Probability": (
                    f"{probability * 100:.2f}%"
                )
            }
        )

    score_df = pd.DataFrame(
        score_rows
    )

    st.dataframe(
        score_df,
        use_container_width=True,
        hide_index=True
    )

    best_score = matrix_result[
        "top_scores"
    ][0]

    # --------------------------------------------------------
    # Prediction summary
    # --------------------------------------------------------

    st.subheader("8. Prediction Summary")

    outcomes = {
        home_team: matrix_result["home_win"],
        "Draw": matrix_result["draw"],
        away_team: matrix_result["away_win"],
    }

    predicted_outcome = max(
        outcomes,
        key=outcomes.get
    )

    predicted_probability = outcomes[
        predicted_outcome
    ]

    score_string = (
        f"{best_score[0]}-{best_score[1]}"
    )

    st.markdown(
        f"""
### Most likely outcome: **{predicted_outcome}**

**Probability:** {predicted_probability * 100:.1f}%

**Most likely score:** **{score_string}**

**Expected goals:**  
{home_team}: **{home_xg:.2f}**  
{away_team}: **{away_xg:.2f}**
"""
    )

    # --------------------------------------------------------
    # Double chance
    # --------------------------------------------------------

    st.subheader("9. Double Chance")

    dc1, dc2, dc3 = st.columns(3)

    with dc1:
        st.metric(
            "1X",
            format_percent(
                matrix_result["home_or_draw"]
            )
        )

    with dc2:
        st.metric(
            "X2",
            format_percent(
                matrix_result["draw_or_away"]
            )
        )

    with dc3:
        st.metric(
            "12",
            format_percent(
                matrix_result["home_or_away"]
            )
        )

    # --------------------------------------------------------
    # Data coverage
    # --------------------------------------------------------

    st.subheader("10. Model Data Coverage")

    d1, d2, d3 = st.columns(3)

    with d1:
        st.metric(
            "Coverage",
            f"{coverage_score}/100"
        )

    with d2:
        st.metric(
            "Coverage Level",
            coverage_level
        )

    with d3:

        if history_model is not None:
            st.metric(
                "Historical Matches",
                f"{history_model['rows']:,}"
            )
        else:
            st.metric(
                "Historical Matches",
                "0"
            )

    if history_model is not None:

        if history_model["using_xg"]:

            st.success(
                "Historical model is using xG data."
            )

        else:

            st.info(
                "Historical model is using actual match goals."
            )

        if history_model["latest_date"]:

            st.caption(
                "Latest historical match: "
                f"{history_model['latest_date']}"
            )

    else:

        st.warning(
            "For stronger predictions, upload a historical "
            "CSV containing previous matches."
        )

    # --------------------------------------------------------
    # Context summary
    # --------------------------------------------------------

    st.subheader("11. Match Context Used")

    context_rows = [
        ["Competition", competition],
        [
            "Season",
            (
                custom_season
                if (
                    season_choice == "Other / Custom"
                    and custom_season
                )
                else season_choice
            )
        ],
        ["Stage", match_stage],
        ["Venue", venue],
        ["Weather", weather],
        ["Pitch", pitch],
        [
            f"{home_team} Formation",
            home_formation
        ],
        [
            f"{away_team} Formation",
            away_formation
        ],
        [
            f"{home_team} Motivation",
            home_motivation
        ],
        [
            f"{away_team} Motivation",
            away_motivation
        ],
        [
            f"{home_team} Absences",
            (
                f"{home_absences} "
                f"({home_absence_type})"
            )
        ],
        [
            f"{away_team} Absences",
            (
                f"{away_absences} "
                f"({away_absence_type})"
            )
        ],
        [
            f"{home_team} Rest",
            f"{home_rest_days} days"
        ],
        [
            f"{away_team} Rest",
            f"{away_rest_days} days"
        ],
    ]

    context_df = pd.DataFrame(
        context_rows,
        columns=[
            "Factor",
            "Value"
        ]
    )

    st.dataframe(
        context_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # Technical details
    # --------------------------------------------------------

    with st.expander(
        "Technical Model Details"
    ):

        technical_df = pd.DataFrame(
            [
                [
                    "Home attack context",
                    f"{home_attack_context:.3f}"
                ],
                [
                    "Home defence context",
                    f"{home_defence_context:.3f}"
                ],
                [
                    "Away attack context",
                    f"{away_attack_context:.3f}"
                ],
                [
                    "Away defence context",
                    f"{away_defence_context:.3f}"
                ],
                [
                    "Dixon-Coles rho",
                    "-0.05"
                ],
                [
                    "Score matrix maximum",
                    str(max_goals)
                ],
                [
                    "Monte Carlo samples",
                    f"{simulation_count:,}"
                ],
                [
                    "Random seed",
                    str(random_seed)
                ],
            ],
            columns=[
                "Parameter",
                "Value"
            ]
        )

        st.dataframe(
            technical_df,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # Prediction JSON
    # --------------------------------------------------------

    prediction_export = {
        "model": "MATCHLAB Pro V2",

        "home_team": home_team,
        "away_team": away_team,

        "competition": competition,

        "season": (
            custom_season
            if (
                season_choice == "Other / Custom"
                and custom_season
            )
            else season_choice
        ),

        "stage": match_stage,

        "venue": venue,

        "expected_goals": {
            "home": round(
                home_xg,
                4
            ),
            "away": round(
                away_xg,
                4
            ),
        },

        "result_probability": {
            "home_win": round(
                matrix_result["home_win"],
                6
            ),
            "draw": round(
                matrix_result["draw"],
                6
            ),
            "away_win": round(
                matrix_result["away_win"],
                6
            ),
        },

        "goal_probability": {
            "over_1_5": round(
                matrix_result["over_15"],
                6
            ),
            "over_2_5": round(
                matrix_result["over_25"],
                6
            ),
            "over_3_5": round(
                matrix_result["over_35"],
                6
            ),
            "btts_yes": round(
                matrix_result["btts_yes"],
                6
            ),
        },

        "most_likely_score": {
            "home_goals": best_score[0],
            "away_goals": best_score[1],
            "probability": round(
                best_score[2],
                6
            ),
        },

        "data_coverage": {
            "score": coverage_score,
            "level": coverage_level,
        },

        "monte_carlo": {
            "samples": simulation_result[
                "samples"
            ],

            "home_win": round(
                simulation_result["home_win"],
                6
            ),

            "draw": round(
                simulation_result["draw"],
                6
            ),

            "away_win": round(
                simulation_result["away_win"],
                6
            ),
        },
    }

    st.download_button(
        "Download Prediction JSON",
        data=json.dumps(
            prediction_export,
            indent=2
        ),
        file_name=(
            f"matchlab_"
            f"{home_key}_vs_"
            f"{away_key}.json"
        ),
        mime="application/json",
        use_container_width=True
    )

    # --------------------------------------------------------
    # Important notice
    # --------------------------------------------------------

    st.warning(
        "MATCHLAB Pro produces probability estimates, not "
        "guaranteed football results. It does not automatically "
        "verify live injuries, confirmed starting lineups, "
        "bookmaker odds, weather feeds, or real-time match data."
    )

    st.caption(
        "MATCHLAB Pro V2 · Historical strength + form + "
        "context + Dixon-Coles probability matrix + Monte Carlo"
    )
