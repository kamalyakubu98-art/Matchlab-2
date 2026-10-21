import math
import pandas as pd
import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MATCHLAB Pro",
    page_icon="⚽",
    layout="wide"
)

st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        color: #9CA3AF;
        margin-bottom: 25px;
    }

    .result-card {
        background: #151B27;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 15px;
    }

    .score {
        font-size: 32px;
        font-weight: bold;
    }

    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# TEAM DATABASE
#
# Attack = expected attacking strength
# Defense = expected goals conceded strength
# Lower defense value = stronger defense
# ============================================================

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


# ============================================================
# DEFAULT VALUES FOR UNKNOWN TEAMS
# ============================================================

DEFAULT_TEAM = {
    "name": "Unknown Team",
    "league": "Unknown",
    "attack": 1.40,
    "defense": 1.40,
    "home_adv": 0.20,
}


# ============================================================
# MATHEMATICAL FUNCTIONS
# ============================================================

def poisson_pmf(goals, expected_goals):

    if expected_goals <= 0:
        return 1.0 if goals == 0 else 0.0

    return (
        math.exp(-expected_goals)
        * (expected_goals ** goals)
        / math.factorial(goals)
    )


def dixon_coles_adjustment(
    home_goals,
    away_goals,
    home_xg,
    away_xg,
    rho=-0.05
):

    if home_goals == 0 and away_goals == 0:
        adjustment = 1 - (home_xg * away_xg * rho)

    elif home_goals == 0 and away_goals == 1:
        adjustment = 1 + (home_xg * rho)

    elif home_goals == 1 and away_goals == 0:
        adjustment = 1 + (away_xg * rho)

    elif home_goals == 1 and away_goals == 1:
        adjustment = 1 - rho

    else:
        adjustment = 1.0

    # Prevent invalid negative probability multipliers
    return max(0.0, adjustment)


# ============================================================
# EXPECTED GOALS MODEL
# ============================================================

def calculate_expected_goals(
    home_team,
    away_team,
    weather_factor=1.0,
    home_form=1.0,
    away_form=1.0,
):

    home = TEAM_DATA.get(home_team, DEFAULT_TEAM)
    away = TEAM_DATA.get(away_team, DEFAULT_TEAM)

    # --------------------------------------------------------
    # Home attack versus away defense
    # --------------------------------------------------------

    home_xg = (
        home["attack"]
        * away["defense"]
        * home_form
        * weather_factor
    )

    # Apply home advantage
    home_xg += home["home_adv"]

    # --------------------------------------------------------
    # Away attack versus home defense
    # --------------------------------------------------------

    away_xg = (
        away["attack"]
        * home["defense"]
        * away_form
        * weather_factor
    )

    # Keep predictions within sensible numerical boundaries
    home_xg = max(0.20, min(home_xg, 4.50))
    away_xg = max(0.20, min(away_xg, 4.50))

    return home_xg, away_xg


# ============================================================
# SCORE MATRIX
# ============================================================

def calculate_probabilities(
    home_xg,
    away_xg,
    max_goals=10,
    rho=-0.05
):

    probabilities = {}

    for home_goals in range(max_goals + 1):

        for away_goals in range(max_goals + 1):

            home_probability = poisson_pmf(
                home_goals,
                home_xg
            )

            away_probability = poisson_pmf(
                away_goals,
                away_xg
            )

            adjustment = dixon_coles_adjustment(
                home_goals,
                away_goals,
                home_xg,
                away_xg,
                rho
            )

            probability = (
                home_probability
                * away_probability
                * adjustment
            )

            probabilities[
                (home_goals, away_goals)
            ] = probability

    # --------------------------------------------------------
    # NORMALIZE DISTRIBUTION
    # --------------------------------------------------------

    total_probability = sum(probabilities.values())

    if total_probability <= 0:
        raise ValueError(
            "Probability distribution could not be calculated."
        )

    probabilities = {
        score: probability / total_probability
        for score, probability in probabilities.items()
    }

    # --------------------------------------------------------
    # 1X2
    # --------------------------------------------------------

    home_win = sum(
        probability
        for (h, a), probability in probabilities.items()
        if h > a
    )

    draw = sum(
        probability
        for (h, a), probability in probabilities.items()
        if h == a
    )

    away_win = sum(
        probability
        for (h, a), probability in probabilities.items()
        if h < a
    )

    # --------------------------------------------------------
    # GOAL MARKETS
    # --------------------------------------------------------

    over_15 = sum(
        p for (h, a), p in probabilities.items()
        if h + a >= 2
    )

    over_25 = sum(
        p for (h, a), p in probabilities.items()
        if h + a >= 3
    )

    over_35 = sum(
        p for (h, a), p in probabilities.items()
        if h + a >= 4
    )

    btts = sum(
        p for (h, a), p in probabilities.items()
        if h > 0 and a > 0
    )

    # --------------------------------------------------------
    # CORRECT SCORES
    #
    # IMPORTANT:
    # Sort by probability, not by score tuple.
    # --------------------------------------------------------

    top_scores = sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True
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
        "matrix": probabilities,
    }


# ============================================================
# FAIR ODDS
# ============================================================

def fair_odds(probability):

    if probability <= 0:
        return 0

    return 1 / probability


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚽ Match Input")

home_input = st.sidebar.text_input(
    "Home Team",
    "Aston Villa"
)

away_input = st.sidebar.text_input(
    "Away Team",
    "Manchester City"
)

st.sidebar.divider()

st.sidebar.subheader("Context Adjustments")

home_form = st.sidebar.slider(
    "Home Form Strength",
    min_value=0.70,
    max_value=1.30,
    value=1.00,
    step=0.01
)

away_form = st.sidebar.slider(
    "Away Form Strength",
    min_value=0.70,
    max_value=1.30,
    value=1.00,
    step=0.01
)

weather_factor = st.sidebar.slider(
    "Weather / Pitch Factor",
    min_value=0.80,
    max_value=1.05,
    value=1.00,
    step=0.01
)


# ============================================================
# NORMALIZE INPUT
# ============================================================

home_key = home_input.strip().lower()
away_key = away_input.strip().lower()

home_data = TEAM_DATA.get(home_key)
away_data = TEAM_DATA.get(away_key)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">⚽ MATCHLAB 4.0</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Statistical Football Probability Engine'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# INPUT VALIDATION
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


if home_data is None:

    st.warning(
        f"No stored ratings found for {home_input}. "
        "League-average ratings are being used."
    )


if away_data is None:

    st.warning(
        f"No stored ratings found for {away_input}. "
        "League-average ratings are being used."
    )


# ============================================================
# RUN MODEL
# ============================================================

try:

    home_xg, away_xg = calculate_expected_goals(
        home_key,
        away_key,
        weather_factor,
        home_form,
        away_form
    )

    results = calculate_probabilities(
        home_xg,
        away_xg
    )

except Exception as error:

    st.error(
        f"Prediction engine error: {error}"
    )

    st.stop()


# ============================================================
# MATCH HEADER
# ============================================================

st.header(
    f"{home_input.title()} vs {away_input.title()}"
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Home Expected Goals",
        f"{home_xg:.2f}"
    )

with col2:

    st.metric(
        "Total Expected Goals",
        f"{home_xg + away_xg:.2f}"
    )

with col3:

    st.metric(
        "Away Expected Goals",
        f"{away_xg:.2f}"
    )


# ============================================================
# 1X2 MARKET
# ============================================================

st.subheader("1X2 Probabilities")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        f"{home_input.title()} Win",
        f"{results['home_win'] * 100:.1f}%"
    )

    st.caption(
        f"Fair odds: "
        f"{fair_odds(results['home_win']):.2f}"
    )

with col2:

    st.metric(
        "Draw",
        f"{results['draw'] * 100:.1f}%"
    )

    st.caption(
        f"Fair odds: "
        f"{fair_odds(results['draw']):.2f}"
    )

with col3:

    st.metric(
        f"{away_input.title()} Win",
        f"{results['away_win'] * 100:.1f}%"
    )

    st.caption(
        f"Fair odds: "
        f"{fair_odds(results['away_win']):.2f}"
    )


# ============================================================
# GOAL MARKETS
# ============================================================

st.subheader("Goal Markets")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Over 1.5",
        f"{results['over_15'] * 100:.1f}%"
    )

with col2:
    st.metric(
        "Over 2.5",
        f"{results['over_25'] * 100:.1f}%"
    )

with col3:
    st.metric(
        "Over 3.5",
        f"{results['over_35'] * 100:.1f}%"
    )

with col4:
    st.metric(
        "BTTS",
        f"{results['btts'] * 100:.1f}%"
    )


# ============================================================
# MODEL PICK
# ============================================================

outcomes = {
    home_input.title(): results["home_win"],
    "Draw": results["draw"],
    away_input.title(): results["away_win"],
}

model_pick = max(
    outcomes,
    key=outcomes.get
)

model_probability = outcomes[model_pick]

best_score, best_score_probability = (
    results["top_scores"][0]
)

st.subheader("Model Verdict")

st.success(
    f"Result lean: {model_pick} "
    f"({model_probability * 100:.1f}%)"
)

st.info(
    f"Most likely correct score: "
    f"{best_score[0]}-{best_score[1]} "
    f"({best_score_probability * 100:.1f}%)"
)


# ============================================================
# TOP CORRECT SCORES
# ============================================================

st.subheader("Most Likely Correct Scores")

score_rows = []

for (home_goals, away_goals), probability in results["top_scores"]:

    score_rows.append({
        "Score": f"{home_goals}-{away_goals}",
        "Probability": f"{probability * 100:.2f}%",
        "Fair Odds": round(
            fair_odds(probability),
            2
        )
    })

score_df = pd.DataFrame(score_rows)

st.dataframe(
    score_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# MARKET SUMMARY
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
        "BTTS Yes"
    ],

    "Probability": [
        results["home_win"],
        results["draw"],
        results["away_win"],
        results["over_15"],
        results["over_25"],
        results["over_35"],
        results["btts"]
    ]
})

summary["Probability"] = (
    summary["Probability"] * 100
).round(1).astype(str) + "%"

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("How this model works"):

    st.write("""
    MATCHLAB estimates expected goals from the attacking
    strength of each team, the opponent's defensive strength,
    home advantage and user-controlled contextual adjustments.

    The expected-goal estimates are converted into score
    probabilities using a Poisson model with a Dixon-Coles
    low-score correction.

    Probabilities are then normalized before calculating
    match-result, goals, BTTS and correct-score markets.

    The displayed odds are mathematical fair odds and do not
    include bookmaker margin.
    """)

st.caption(
    "MATCHLAB produces statistical estimates, not guaranteed outcomes."
)
