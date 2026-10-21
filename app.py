import math
import pandas as pd
import streamlit as st

# ============================================================
# MATCHLAB 4.1 — Statistical Football Probability Engine
# Run with: streamlit run app.py
# ============================================================

st.set_page_config(
    page_title="MATCHLAB Pro",
    page_icon="⚽",
    layout="wide",
)

st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0;
    }
    .subtitle {
        color: #9CA3AF;
        margin-bottom: 25px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# TEAM RATINGS
# These are illustrative manual ratings, not live statistics.
# Attack approximates scoring strength.
# Defense approximates goals-conceded factor; lower is stronger.
# ------------------------------------------------------------

TEAM_DATA = {
    "aston villa": {
        "name": "Aston Villa",
        "league": "Premier League",
        "attack": 1.65,
        "defense": 1.20,
        "home_adv": 0.25,
    },
    "manchester city": {
        "name": "Manchester City",
        "league": "Premier League",
        "attack": 2.30,
        "defense": 0.90,
        "home_adv": 0.30,
    },
    "arsenal": {
        "name": "Arsenal",
        "league": "Premier League",
        "attack": 2.10,
        "defense": 0.85,
        "home_adv": 0.22,
    },
    "liverpool": {
        "name": "Liverpool",
        "league": "Premier League",
        "attack": 2.25,
        "defense": 0.95,
        "home_adv": 0.35,
    },
    "chelsea": {
        "name": "Chelsea",
        "league": "Premier League",
        "attack": 1.80,
        "defense": 1.35,
        "home_adv": 0.15,
    },
    "manchester united": {
        "name": "Manchester United",
        "league": "Premier League",
        "attack": 1.45,
        "defense": 1.40,
        "home_adv": 0.10,
    },
    "real madrid": {
        "name": "Real Madrid",
        "league": "La Liga",
        "attack": 2.40,
        "defense": 0.88,
        "home_adv": 0.32,
    },
    "barcelona": {
        "name": "Barcelona",
        "league": "La Liga",
        "attack": 2.05,
        "defense": 1.05,
        "home_adv": 0.28,
    },
    "bayern munich": {
        "name": "Bayern Munich",
        "league": "Bundesliga",
        "attack": 2.35,
        "defense": 1.10,
        "home_adv": 0.30,
    },
    "inter milan": {
        "name": "Inter Milan",
        "league": "Serie A",
        "attack": 1.95,
        "defense": 0.80,
        "home_adv": 0.25,
    },
    "juventus": {
        "name": "Juventus",
        "league": "Serie A",
        "attack": 1.60,
        "defense": 0.90,
        "home_adv": 0.18,
    },
    "psg": {
        "name": "PSG",
        "league": "Ligue 1",
        "attack": 2.15,
        "defense": 1.15,
        "home_adv": 0.20,
    },
}

# ------------------------------------------------------------
# COMMON TEAM NAME ALIASES
# ------------------------------------------------------------

ALIASES = {
    "man city": "manchester city",
    "manchester city fc": "manchester city",
    "man utd": "manchester united",
    "man united": "manchester united",
    "manchester united fc": "manchester united",
    "real": "real madrid",
    "barca": "barcelona",
    "bayern": "bayern munich",
    "inter": "inter milan",
    "paris saint-germain": "psg",
    "paris saint germain": "psg",
}

DEFAULT_TEAM = {
    "name": "Unknown Team",
    "league": "Unknown",
    "attack": 1.40,
    "defense": 1.40,
    "home_adv": 0.20,
}


def normalize_team_name(name: str) -> str:
    key = " ".join(name.strip().lower().split())
    return ALIASES.get(key, key)


def poisson_pmf(goals: int, expected_goals: float) -> float:
    """Calculate the Poisson probability of an exact goal count."""

    if goals < 0 or expected_goals < 0:
        return 0.0

    if expected_goals == 0:
        return 1.0 if goals == 0 else 0.0

    return math.exp(
        -expected_goals
        + goals * math.log(expected_goals)
        - math.lgamma(goals + 1)
    )


def dixon_coles_adjustment(
    home_goals: int,
    away_goals: int,
    home_xg: float,
    away_xg: float,
    rho: float = -0.05,
) -> float:
    """Adjust probabilities for low-scoring results."""

    if home_goals == 0 and away_goals == 0:
        tau = 1 - home_xg * away_xg * rho

    elif home_goals == 0 and away_goals == 1:
        tau = 1 + home_xg * rho

    elif home_goals == 1 and away_goals == 0:
        tau = 1 + away_xg * rho

    elif home_goals == 1 and away_goals == 1:
        tau = 1 - rho

    else:
        tau = 1.0

    return max(0.0, tau)


def calculate_expected_goals(
    home_team: str,
    away_team: str,
    weather_factor: float = 1.0,
    home_form: float = 1.0,
    away_form: float = 1.0,
):
    """Estimate each team's expected goals."""

    home = TEAM_DATA.get(
        normalize_team_name(home_team),
        DEFAULT_TEAM,
    )

    away = TEAM_DATA.get(
        normalize_team_name(away_team),
        DEFAULT_TEAM,
    )

    home_xg = (
        home["attack"]
        * away["defense"]
        * home_form
        * weather_factor
    )

    home_xg += home["home_adv"]

    away_xg = (
        away["attack"]
        * home["defense"]
        * away_form
        * weather_factor
    )

    home_xg = max(0.05, min(home_xg, 5.0))
    away_xg = max(0.05, min(away_xg, 5.0))

    return home_xg, away_xg


def calculate_probabilities(
    home_xg: float,
    away_xg: float,
    max_goals: int = 10,
    rho: float = -0.05,
):
    """Calculate match, goals-market and correct-score probabilities."""

    if not (
        math.isfinite(home_xg)
        and math.isfinite(away_xg)
    ):
        raise ValueError(
            "Expected goals must be finite numbers."
        )

    if home_xg < 0 or away_xg < 0:
        raise ValueError(
            "Expected goals cannot be negative."
        )

    if max_goals < 5:
        raise ValueError(
            "Maximum goals must be at least 5."
        )

    raw = {}

    for h in range(max_goals + 1):
        ph = poisson_pmf(h, home_xg)

        for a in range(max_goals + 1):
            pa = poisson_pmf(a, away_xg)

            tau = dixon_coles_adjustment(
                h,
                a,
                home_xg,
                away_xg,
                rho,
            )

            raw[(h, a)] = ph * pa * tau

    raw_total = sum(raw.values())

    if (
        not math.isfinite(raw_total)
        or raw_total <= 0
    ):
        raise ValueError(
            "Could not calculate a valid score distribution."
        )

    # Estimate probability mass outside the selected score range.
    home_mass = sum(
        poisson_pmf(g, home_xg)
        for g in range(max_goals + 1)
    )

    away_mass = sum(
        poisson_pmf(g, away_xg)
        for g in range(max_goals + 1)
    )

    omitted_mass = max(
        0.0,
        1.0 - home_mass * away_mass,
    )

    # Normalize the finite score matrix.
    matrix = {
        score: probability / raw_total
        for score, probability in raw.items()
    }

    home_win = sum(
        p for (h, a), p in matrix.items()
        if h > a
    )

    draw = sum(
        p for (h, a), p in matrix.items()
        if h == a
    )

    away_win = sum(
        p for (h, a), p in matrix.items()
        if h < a
    )

    over_15 = sum(
        p for (h, a), p in matrix.items()
        if h + a >= 2
    )

    over_25 = sum(
        p for (h, a), p in matrix.items()
        if h + a >= 3
    )

    over_35 = sum(
        p for (h, a), p in matrix.items()
        if h + a >= 4
    )

    btts = sum(
        p for (h, a), p in matrix.items()
        if h > 0 and a > 0
    )

    top_scores = sorted(
        matrix.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:10]

    return {
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "over_15": over_15,
        "over_25": over_25,
        "over_35": over_35,
        "btts": btts,
        "top_scores": top_scores,
        "matrix": matrix,
        "omitted_mass_estimate": omitted_mass,
        "result_probability_sum": (
            home_win + draw + away_win
        ),
    }


def fair_odds(probability: float) -> float:
    """Calculate theoretical odds without bookmaker margin."""

    if probability <= 0:
        return float("inf")

    return 1.0 / probability


def confidence_label(probability: float) -> str:
    """Describe the model's probability band."""

    if probability >= 0.70:
        return "High model probability"

    if probability >= 0.55:
        return "Moderate model probability"

    return "Low model probability"


# ============================================================
# SIDEBAR — MATCH INPUT
# ============================================================

st.sidebar.title("⚽ Match Input")

home_input = st.sidebar.text_input(
    "Home Team",
    "Aston Villa",
)

away_input = st.sidebar.text_input(
    "Away Team",
    "Manchester City",
)

st.sidebar.divider()
st.sidebar.subheader("Context Adjustments")

home_form = st.sidebar.slider(
    "Home Form Strength",
    0.70,
    1.30,
    1.00,
    0.01,
)

away_form = st.sidebar.slider(
    "Away Form Strength",
    0.70,
    1.30,
    1.00,
    0.01,
)

weather_factor = st.sidebar.slider(
    "Weather / Pitch Factor",
    0.80,
    1.05,
    1.00,
    0.01,
)

max_goals = st.sidebar.selectbox(
    "Maximum goals per team in score matrix",
    [8, 10, 12, 15],
    index=1,
)

home_key = normalize_team_name(home_input)
away_key = normalize_team_name(away_input)

st.markdown(
    '<div class="main-title">⚽ MATCHLAB 4.1</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Statistical Football Probability Engine'
    '</div>',
    unsafe_allow_html=True,
)

# ============================================================
# VALIDATION
# ============================================================

if not home_input.strip() or not away_input.strip():
    st.error(
        "Please enter both the home and away teams."
    )
    st.stop()

if home_key == away_key:
    st.error(
        "Home and away teams cannot be the same."
    )
    st.stop()

home_data = TEAM_DATA.get(home_key)
away_data = TEAM_DATA.get(away_key)

if home_data is None:
    st.warning(
        f"No stored ratings found for "
        f"{home_input.strip()}. Illustrative default "
        f"ratings are being used. This is not a "
        f"team-specific forecast."
    )

if away_data is None:
    st.warning(
        f"No stored ratings found for "
        f"{away_input.strip()}. Illustrative default "
        f"ratings are being used. This is not a "
        f"team-specific forecast."
    )

try:
    home_xg, away_xg = calculate_expected_goals(
        home_key,
        away_key,
        weather_factor,
        home_form,
        away_form,
    )

    results = calculate_probabilities(
        home_xg,
        away_xg,
        max_goals=max_goals,
    )

except (
    ValueError,
    OverflowError,
    ZeroDivisionError,
) as error:
    st.error(f"Prediction engine error: {error}")
    st.stop()

home_display = (
    home_data["name"]
    if home_data
    else home_input.strip().title()
)

away_display = (
    away_data["name"]
    if away_data
    else away_input.strip().title()
)

# ============================================================
# EXPECTED GOALS
# ============================================================

st.header(
    f"{home_display} vs {away_display}"
)

c1, c2, c3 = st.columns(3)

c1.metric(
    "Home Expected Goals",
    f"{home_xg:.2f}",
)

c2.metric(
    "Total Expected Goals",
    f"{home_xg + away_xg:.2f}",
)

c3.metric(
    "Away Expected Goals",
    f"{away_xg:.2f}",
)

# ============================================================
# 1X2 PROBABILITIES
# ============================================================

st.subheader("1X2 Probabilities")

c1, c2, c3 = st.columns(3)

for col, label, key in [
    (
        c1,
        f"{home_display} Win",
        "home_win",
    ),
    (
        c2,
        "Draw",
        "draw",
    ),
    (
        c3,
        f"{away_display} Win",
        "away_win",
    ),
]:
    probability = results[key]

    col.metric(
        label,
        f"{probability * 100:.1f}%",
    )

    col.caption(
        f"Fair odds: {fair_odds(probability):.2f}"
    )

# ============================================================
# GOALS MARKETS
# ============================================================

st.subheader("Goal Markets")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Over 1.5",
    f"{results['over_15'] * 100:.1f}%",
)

c2.metric(
    "Over 2.5",
    f"{results['over_25'] * 100:.1f}%",
)

c3.metric(
    "Over 3.5",
    f"{results['over_35'] * 100:.1f}%",
)

c4.metric(
    "BTTS — Yes",
    f"{results['btts'] * 100:.1f}%",
)

# ============================================================
# MODEL VERDICT
# CORRECTED OUTCOMES DICTIONARY
# ============================================================

outcomes = {
    home_display: results["home_win"],
    "Draw": results["draw"],
    away_display: results["away_win"],
}

model_pick = max(
    outcomes,
    key=outcomes.get,
)

model_probability = outcomes[model_pick]

best_score, best_score_probability = (
    results["top_scores"][0]
)

st.subheader("Model Verdict")

st.success(
    f"Most probable 1X2 outcome: "
    f"{model_pick} "
    f"({model_probability * 100:.1f}%)"
)

st.caption(
    confidence_label(model_probability)
    + " — this is a probability band, "
    "not a guarantee of accuracy."
)

st.info(
    f"Most likely correct score: "
    f"{best_score[0]}-{best_score[1]} "
    f"({best_score_probability * 100:.2f}%)"
)

# ============================================================
# MOST LIKELY CORRECT SCORES
# ============================================================

st.subheader("Most Likely Correct Scores")

score_rows = [
    {
        "Score": f"{h}-{a}",
        "Probability": f"{p * 100:.2f}%",
        "Fair Odds": round(fair_odds(p), 2),
    }
    for (h, a), p in results["top_scores"]
]

st.dataframe(
    pd.DataFrame(score_rows),
    use_container_width=True,
    hide_index=True,
)

# ============================================================
# PREDICTION SUMMARY
# ============================================================

st.subheader("Prediction Summary")

summary = pd.DataFrame({
    "Market": [
        "Home Win",
        "Draw",
        "Away Win",
        "Over 1.5",
        "Over 2.5",
        "Over 3.5",
        "BTTS Yes",
    ],
    "Probability": [
        results["home_win"],
        results["draw"],
        results["away_win"],
        results["over_15"],
        results["over_25"],
        results["over_35"],
        results["btts"],
    ],
})

summary["Probability"] = (
    summary["Probability"] * 100
).round(1).astype(str) + "%"

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True,
)

# ============================================================
# MODEL CHECKS AND LIMITATIONS
# ============================================================

with st.expander("Model checks and limitations"):

    st.write(
        "1X2 probability sum: "
        f"{results['result_probability_sum'] * 100:.6f}%"
    )

    st.write(
        "Estimated independent Poisson mass beyond "
        "the selected score limit: "
        f"{results['omitted_mass_estimate'] * 100:.6f}%"
    )

    st.write("""
    The engine combines manually supplied attacking and
    defensive ratings, home advantage, form multipliers,
    and a weather/pitch multiplier.

    It then applies independent Poisson goal probabilities
    with a Dixon-Coles low-score correction and normalizes
    the finite score matrix.

    Team ratings are illustrative and are not automatically
    updated from live results.

    Form and weather controls are assumptions entered by
    the user. They are not verified historical statistics.

    To assess predictive quality, fit ratings from verified
    historical matches and run a chronological backtest
    using metrics such as log loss and Brier score.

    Displayed fair odds are the reciprocal of model
    probabilities and exclude bookmaker margin.
    """)

st.caption(
    "MATCHLAB provides statistical estimates, "
    "not guaranteed outcomes."
)
