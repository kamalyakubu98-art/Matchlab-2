import math
import pandas as pd
import streamlit as st

# ============================================================
# MATCHLAB 4.2 — Context-aware football probability engine
# Run with: streamlit run app.py
# NOTE: Ratings and context effects are heuristic unless replaced
# with validated historical data. No live data is fetched here.
# ============================================================

st.set_page_config(page_title="MATCHLAB Pro", page_icon="⚽", layout="wide")

st.markdown("""
<style>
    .main-title {font-size: 42px; font-weight: 800; margin-bottom: 0;}
    .subtitle {color: #9CA3AF; margin-bottom: 18px;}
    .welcome {padding: 18px 22px; border: 1px solid rgba(128,128,128,.35); border-radius: 14px; margin: 8px 0 24px 0;}
    .welcome h2 {margin: 0 0 6px 0;}
    div[data-testid="stMetricValue"] {font-size: 28px; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# Ratings are illustrative starting values, not verified 2026/27 season statistics.
TEAM_DATA = {
    "aston villa": {"name": "Aston Villa", "league": "Premier League", "attack": 1.65, "defense": 1.20, "home_adv": 0.25},
    "manchester city": {"name": "Manchester City", "league": "Premier League", "attack": 2.30, "defense": 0.90, "home_adv": 0.30},
    "arsenal": {"name": "Arsenal", "league": "Premier League", "attack": 2.10, "defense": 0.85, "home_adv": 0.22},
    "liverpool": {"name": "Liverpool", "league": "Premier League", "attack": 2.25, "defense": 0.95, "home_adv": 0.35},
    "chelsea": {"name": "Chelsea", "league": "Premier League", "attack": 1.80, "defense": 1.35, "home_adv": 0.15},
    "manchester united": {"name": "Manchester United", "league": "Premier League", "attack": 1.45, "defense": 1.40, "home_adv": 0.10},
    "real madrid": {"name": "Real Madrid", "league": "La Liga", "attack": 2.40, "defense": 0.88, "home_adv": 0.32},
    "barcelona": {"name": "Barcelona", "league": "La Liga", "attack": 2.05, "defense": 1.05, "home_adv": 0.28},
    "bayern munich": {"name": "Bayern Munich", "league": "Bundesliga", "attack": 2.35, "defense": 1.10, "home_adv": 0.30},
    "inter milan": {"name": "Inter Milan", "league": "Serie A", "attack": 1.95, "defense": 0.80, "home_adv": 0.25},
    "juventus": {"name": "Juventus", "league": "Serie A", "attack": 1.60, "defense": 0.90, "home_adv": 0.18},
    "psg": {"name": "PSG", "league": "Ligue 1", "attack": 2.15, "defense": 1.15, "home_adv": 0.20},
}
ALIASES = {
    "man city": "manchester city", "manchester city fc": "manchester city",
    "man utd": "manchester united", "man united": "manchester united",
    "manchester united fc": "manchester united", "real": "real madrid",
    "barca": "barcelona", "bayern": "bayern munich", "inter": "inter milan",
    "paris saint-germain": "psg", "paris saint germain": "psg",
}
DEFAULT_TEAM = {"name": "Unknown Team", "league": "Unknown", "attack": 1.40, "defense": 1.40, "home_adv": 0.20}


def normalize_team_name(name: str) -> str:
    key = " ".join(name.strip().lower().split())
    return ALIASES.get(key, key)


def poisson_pmf(goals: int, expected_goals: float) -> float:
    if goals < 0 or expected_goals < 0:
        return 0.0
    if expected_goals == 0:
        return 1.0 if goals == 0 else 0.0
    return math.exp(-expected_goals + goals * math.log(expected_goals) - math.lgamma(goals + 1))


def dixon_coles_adjustment(home_goals: int, away_goals: int, home_xg: float, away_xg: float, rho: float = -0.05) -> float:
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


# Effects below are intentionally modest heuristics, not empirically validated constants.
# Form percentage is interpreted as the share of the team's baseline rating retained.
FORMATION_ATTACK = {
    "4-3-3": 0.02, "4-2-3-1": 0.01, "4-4-2": 0.00,
    "3-5-2": 0.01, "3-4-3": 0.02, "4-1-4-1": 0.00,
    "5-3-2": -0.02, "5-4-1": -0.03, "Other / not specified": 0.00,
}
FORMATION_DEFENCE = {
    "4-3-3": 0.00, "4-2-3-1": 0.01, "4-4-2": 0.00,
    "3-5-2": 0.01, "3-4-3": -0.01, "4-1-4-1": 0.02,
    "5-3-2": 0.02, "5-4-1": 0.03, "Other / not specified": 0.00,
}
WEATHER_EFFECT = {
    "Clear / Sunny": 0.00, "Cloudy": 0.00, "Light rain": -0.01,
    "Heavy rain / waterlogged pitch": -0.05, "Strong wind": -0.04,
    "Fog": 0.00, "Thunderstorm / unsafe conditions": -0.08,
    "Snow / icy pitch": -0.06,
}
MOTIVATION_EFFECT = {"Not specified / normal": 0.00, "Low": -0.015, "High": 0.015, "Very high": 0.025}


def calculate_expected_goals(home_team: str, away_team: str, weather_effect: float = 0.0,
                             home_form: float = 1.0, away_form: float = 1.0,
                             home_formation: str = "Other / not specified",
                             away_formation: str = "Other / not specified",
                             home_motivation: str = "Not specified / normal",
                             away_motivation: str = "Not specified / normal",
                             home_missing_attackers: int = 0, away_missing_attackers: int = 0,
                             home_missing_defenders: int = 0, away_missing_defenders: int = 0,
                             home_missing_other: int = 0, away_missing_other: int = 0,
                             home_absence_impact: str = "Regular starter",
                             away_absence_impact: str = "Regular starter"):
    home = TEAM_DATA.get(normalize_team_name(home_team), DEFAULT_TEAM)
    away = TEAM_DATA.get(normalize_team_name(away_team), DEFAULT_TEAM)

    # Blend the form slider with baseline strength instead of multiplying the entire model blindly.
    # 100% retains the baseline; lower values temper the baseline contribution toward neutral.
    neutral_attack = 1.40
    home_attack = neutral_attack + (home["attack"] - neutral_attack) * home_form
    away_attack = neutral_attack + (away["attack"] - neutral_attack) * away_form

    # Formation offsets are tiny priors, not claims that one shape is inherently superior.
    home_xg = home_attack * away["defense"] + home["home_adv"]
    away_xg = away_attack * home["defense"]
    home_xg *= 1.0 + FORMATION_ATTACK[home_formation] + MOTIVATION_EFFECT[home_motivation]
    away_xg *= 1.0 + FORMATION_ATTACK[away_formation] + MOTIVATION_EFFECT[away_motivation]
    # The opponent's shape has a small defensive prior against this team's attack.
    home_xg *= max(0.90, 1.0 - FORMATION_DEFENCE[away_formation])
    away_xg *= max(0.90, 1.0 - FORMATION_DEFENCE[home_formation])

    # Absence effects use counts and a user-selected impact level because names alone cannot
    # establish player value without player-level data. Caps prevent extreme manual inputs.
    impact = {"Squad player": 0.025, "Regular starter": 0.06, "Key player": 0.10}
    h_imp = impact[home_absence_impact]
    a_imp = impact[away_absence_impact]
    home_xg *= max(0.65, 1.0 - min(3, home_missing_attackers) * h_imp - min(3, home_missing_other) * h_imp * 0.25)
    away_xg *= max(0.65, 1.0 - min(3, away_missing_attackers) * a_imp - min(3, away_missing_other) * a_imp * 0.25)

    # Missing defenders raise the opponent's expected goals, rather than the affected team's.
    home_defence_vulnerability = min(0.25, min(3, home_missing_defenders) * h_imp)
    away_defence_vulnerability = min(0.25, min(3, away_missing_defenders) * a_imp)
    away_xg *= 1.0 + home_defence_vulnerability
    home_xg *= 1.0 + away_defence_vulnerability

    # Weather affects both sides slightly; only severe conditions receive a larger prior.
    home_xg *= 1.0 + weather_effect
    away_xg *= 1.0 + weather_effect
    return max(0.05, min(home_xg, 5.0)), max(0.05, min(away_xg, 5.0))


def calculate_probabilities(home_xg: float, away_xg: float, max_goals: int = 8, rho: float = -0.05):
    if not (math.isfinite(home_xg) and math.isfinite(away_xg)):
        raise ValueError("Expected goals must be finite numbers.")
    if home_xg < 0 or away_xg < 0:
        raise ValueError("Expected goals cannot be negative.")
    if max_goals < 5:
        raise ValueError("max_goals must be at least 5.")
    raw = {}
    for h in range(max_goals + 1):
        ph = poisson_pmf(h, home_xg)
        for a in range(max_goals + 1):
            pa = poisson_pmf(a, away_xg)
            raw[(h, a)] = ph * pa * dixon_coles_adjustment(h, a, home_xg, away_xg, rho)
    raw_total = sum(raw.values())
    if not math.isfinite(raw_total) or raw_total <= 0:
        raise ValueError("Could not calculate a valid score probability distribution.")
    mass_home = sum(poisson_pmf(g, home_xg) for g in range(max_goals + 1))
    mass_away = sum(poisson_pmf(g, away_xg) for g in range(max_goals + 1))
    omitted_mass = max(0.0, 1.0 - mass_home * mass_away)
    matrix = {score: p / raw_total for score, p in raw.items()}
    home_win = sum(p for (h, a), p in matrix.items() if h > a)
    draw = sum(p for (h, a), p in matrix.items() if h == a)
    away_win = sum(p for (h, a), p in matrix.items() if h < a)
    over_15 = sum(p for (h, a), p in matrix.items() if h + a >= 2)
    over_25 = sum(p for (h, a), p in matrix.items() if h + a >= 3)
    over_35 = sum(p for (h, a), p in matrix.items() if h + a >= 4)
    btts = sum(p for (h, a), p in matrix.items() if h > 0 and a > 0)
    top_scores = sorted(matrix.items(), key=lambda item: item[1], reverse=True)[:10]
    return {"home_win": home_win, "draw": draw, "away_win": away_win,
            "over_15": over_15, "over_25": over_25, "over_35": over_35,
            "btts": btts, "top_scores": top_scores, "matrix": matrix,
            "omitted_mass_estimate": omitted_mass,
            "result_probability_sum": home_win + draw + away_win}


def fair_odds(probability: float):
    return 1.0 / probability if probability > 0 else float("inf")


def confidence_label(probability: float) -> str:
    if probability >= 0.70:
        return "High model probability"
    if probability >= 0.55:
        return "Moderate model probability"
    return "Low model probability"


# ------------------------------ SIDEBAR ------------------------------
st.sidebar.title("⚽ Match Input")
st.sidebar.caption("Season: 2026/2027")
home_input = st.sidebar.text_input("Home Team", "Aston Villa")
away_input = st.sidebar.text_input("Away Team", "Manchester City")

st.sidebar.divider()
st.sidebar.subheader("Context Adjustments")
st.sidebar.caption("These optional manual inputs change the model's expected goals. They are heuristic until calibrated against historical data.")

home_form_pct = st.sidebar.slider("Home Form Strength (%)", 5, 100, 100, 5,
    help="How much of the stored team-strength difference from a neutral baseline to retain. This is not a measured recent-form statistic unless you calibrate it.")
away_form_pct = st.sidebar.slider("Away Form Strength (%)", 5, 100, 100, 5,
    help="How much of the stored team-strength difference from a neutral baseline to retain. This is not a measured recent-form statistic unless you calibrate it.")
home_form = home_form_pct / 100.0
away_form = away_form_pct / 100.0

weather_condition = st.sidebar.selectbox("Weather / Pitch Conditions", list(WEATHER_EFFECT.keys()), index=0,
    help="Select only if you have a reasonable expectation of match conditions. Weather effects are small heuristic adjustments, not live weather observations.")
weather_effect = WEATHER_EFFECT[weather_condition]

max_goals = st.sidebar.selectbox("Scoreline Calculation Limit (goals per team)", [5, 6, 7, 8, 9, 10], index=3,
    help="Maximum goals included for each team in the score matrix. A larger limit captures more extreme scorelines; probabilities are normalized over the selected matrix.")

with st.sidebar.expander("Optional: Formations"):
    formations = list(FORMATION_ATTACK.keys())
    home_formation = st.selectbox("Home formation", formations, index=len(formations)-1)
    away_formation = st.selectbox("Away formation", formations, index=len(formations)-1)
    st.caption("Formation effects are deliberately small generic priors. The model does not analyse individual players or tactical matchups.")

with st.sidebar.expander("Optional: Injuries / Suspensions"):
    st.caption("Enter absences you have checked. Names are for your notes; without player-level data, the model uses counts and your impact selection.")
    home_absence_names = st.text_input("Home unavailable players (optional)", placeholder="Player names, separated by commas")
    away_absence_names = st.text_input("Away unavailable players (optional)", placeholder="Player names, separated by commas")
    home_absence_impact = st.selectbox("Impact level — home absences", ["Squad player", "Regular starter", "Key player"], index=1)
    away_absence_impact = st.selectbox("Impact level — away absences", ["Squad player", "Regular starter", "Key player"], index=1)
    home_missing_attackers = st.number_input("Home attackers unavailable", min_value=0, max_value=3, value=0, step=1)
    home_missing_defenders = st.number_input("Home defenders unavailable", min_value=0, max_value=3, value=0, step=1)
    home_missing_other = st.number_input("Home other starters unavailable", min_value=0, max_value=3, value=0, step=1)
    away_missing_attackers = st.number_input("Away attackers unavailable", min_value=0, max_value=3, value=0, step=1)
    away_missing_defenders = st.number_input("Away defenders unavailable", min_value=0, max_value=3, value=0, step=1)
    away_missing_other = st.number_input("Away other starters unavailable", min_value=0, max_value=3, value=0, step=1)

with st.sidebar.expander("Optional: Motivation"):
    motivation_options = list(MOTIVATION_EFFECT.keys())
    home_motivation = st.selectbox("Home motivation", motivation_options, index=0)
    away_motivation = st.selectbox("Away motivation", motivation_options, index=0)
    st.caption("Motivation is subjective and is not directly observable. Keep it at normal unless you have a clear, evidence-based reason to adjust it.")

home_key = normalize_team_name(home_input)
away_key = normalize_team_name(away_input)

st.markdown('<div class="main-title">⚽ MATCHLAB 4.2</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">2026/2027 • Statistical Football Probability Engine</div>', unsafe_allow_html=True)
st.markdown("<div class=\"welcome\"><h2>Welcome to MATCHLAB</h2><p>Input your data to begin the match analysis and simulation.</p></div>", unsafe_allow_html=True)

if not home_input.strip() or not away_input.strip():
    st.error("Please enter both the home and away teams.")
    st.stop()
if home_key == away_key:
    st.error("Home and away teams cannot be the same.")
    st.stop()

home_data = TEAM_DATA.get(home_key)
away_data = TEAM_DATA.get(away_key)
if home_data is None:
    st.warning(f"No stored ratings found for {home_input.strip()}. Generic default ratings are being used; this is not a team-specific forecast.")
if away_data is None:
    st.warning(f"No stored ratings found for {away_input.strip()}. Generic default ratings are being used; this is not a team-specific forecast.")

try:
    home_xg, away_xg = calculate_expected_goals(
        home_key, away_key, weather_effect, home_form, away_form,
        home_formation, away_formation, home_motivation, away_motivation,
        int(home_missing_attackers), int(away_missing_attackers),
        int(home_missing_defenders), int(away_missing_defenders),
        int(home_missing_other), int(away_missing_other),
        home_absence_impact, away_absence_impact,
    )
    results = calculate_probabilities(home_xg, away_xg, max_goals=max_goals)
except (ValueError, OverflowError, ZeroDivisionError) as error:
    st.error(f"Prediction engine error: {error}")
    st.stop()

home_display = home_data["name"] if home_data else home_input.strip().title()
away_display = away_data["name"] if away_data else away_input.strip().title()
st.header(f"{home_display} vs {away_display}")

# Transparent summary of the user-entered context.
context_used = [f"Weather: {weather_condition}", f"Form influence: home {home_form_pct}%, away {away_form_pct}%"]
if home_formation != "Other / not specified" or away_formation != "Other / not specified":
    context_used.append(f"Formation: {home_display} {home_formation}; {away_display} {away_formation}")
if any([home_missing_attackers, home_missing_defenders, home_missing_other, away_missing_attackers, away_missing_defenders, away_missing_other]):
    context_used.append("Absence counts and impact levels applied")
if home_motivation != "Not specified / normal" or away_motivation != "Not specified / normal":
    context_used.append(f"Motivation: home {home_motivation}; away {away_motivation}")
st.caption("Context applied: " + " | ".join(context_used))

c1, c2, c3 = st.columns(3)
c1.metric("Home Expected Goals", f"{home_xg:.2f}")
c2.metric("Total Expected Goals", f"{home_xg + away_xg:.2f}")
c3.metric("Away Expected Goals", f"{away_xg:.2f}")

st.subheader("1X2 Probabilities")
c1, c2, c3 = st.columns(3)
for col, label, key in [(c1, f"{home_display} Win", "home_win"), (c2, "Draw", "draw"), (c3, f"{away_display} Win", "away_win")]:
    col.metric(label, f"{results[key] * 100:.1f}%")
    col.caption(f"Fair odds: {fair_odds(results[key]):.2f}")

st.subheader("Goal Markets")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Over 1.5", f"{results['over_15'] * 100:.1f}%")
c2.metric("Over 2.5", f"{results['over_25'] * 100:.1f}%")
c3.metric("Over 3.5", f"{results['over_35'] * 100:.1f}%")
c4.metric("BTTS — Yes", f"{results['btts'] * 100:.1f}%")

outcomes = {home_display: results["home_win"], "Draw": results["draw"], away_display: results["away_win"]}
model_pick = max(outcomes, key=outcomes.get)
model_probability = outcomes[model_pick]
best_score, best_score_probability = results["top_scores"][0]
st.subheader("Model Verdict")
st.success(f"Most probable 1X2 outcome: {model_pick} ({model_probability * 100:.1f}%)")
st.caption(confidence_label(model_probability) + " — this is a probability band, not a guarantee of predictive accuracy.")
st.info(f"Most likely correct score: {best_score[0]}-{best_score[1]} ({best_score_probability * 100:.2f}%)")

st.subheader("Most Likely Correct Scores")
score_rows = [{"Score": f"{h}-{a}", "Probability": f"{p * 100:.2f}%", "Fair Odds": round(fair_odds(p), 2)} for (h, a), p in results["top_scores"]]
st.dataframe(pd.DataFrame(score_rows), use_container_width=True, hide_index=True)

st.subheader("Prediction Summary")
summary = pd.DataFrame({"Market": ["Home Win", "Draw", "Away Win", "Over 1.5", "Over 2.5", "Over 3.5", "BTTS Yes"],
    "Probability": [results["home_win"], results["draw"], results["away_win"], results["over_15"], results["over_25"], results["over_35"], results["btts"]]})
summary["Probability"] = (summary["Probability"] * 100).round(1).astype(str) + "%"
st.dataframe(summary, use_container_width=True, hide_index=True)

with st.expander("Model checks and limitations"):
    st.write(f"1X2 probability sum: {results['result_probability_sum'] * 100:.6f}%")
    st.write(f"Estimated independent Poisson mass beyond the selected score limit: {results['omitted_mass_estimate'] * 100:.6f}%")
    st.write("""
    MATCHLAB uses manually entered team ratings and a statistical
    Poisson model with a Dixon-Coles low-score correction.

    Form, formation, injury and motivation adjustments use
    heuristic estimates and have not been calibrated against
    historical match data.

    Optional context fields are not verified automatically.
    No live fixtures, weather, injuries, lineups or player
    statistics are retrieved unless external data sources
    are connected.

    The model provides probability estimates, not guaranteed
    outcomes. Historical backtesting is required to evaluate
    predictive accuracy.
    """)

st.caption("MATCHLAB provides statistical estimates, not guaranteed outcomes.")
