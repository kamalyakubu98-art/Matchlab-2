import math
import random
from collections import Counter

import pandas as pd
import streamlit as st

st.set_page_config(page_title="MATCHLAB Pro", page_icon="⚽", layout="wide")

st.markdown("""
<style>
.main-title {font-size: 2.6rem; font-weight: 850; margin-bottom: 0;}
.subtitle {color: #9CA3AF; margin-top: .15rem; margin-bottom: 1.2rem;}
.welcome {padding: 1.25rem 1.4rem; border: 1px solid rgba(128,128,128,.35); border-radius: 16px; margin: .5rem 0 1.3rem 0;}
.small-note {color: #9CA3AF; font-size: .88rem;}
div[data-testid="stMetricValue"] {font-weight: 750;}
</style>
""", unsafe_allow_html=True)

# Illustrative starting ratings only. Replace with verified, season-specific data for real forecasts.
TEAM_DATA = {
    "aston villa": {"name": "Aston Villa", "attack": 1.65, "defense": 1.20, "home_adv": 0.25},
    "manchester city": {"name": "Manchester City", "attack": 2.30, "defense": 0.90, "home_adv": 0.30},
    "arsenal": {"name": "Arsenal", "attack": 2.10, "defense": 0.85, "home_adv": 0.22},
    "liverpool": {"name": "Liverpool", "attack": 2.25, "defense": 0.95, "home_adv": 0.35},
    "chelsea": {"name": "Chelsea", "attack": 1.80, "defense": 1.35, "home_adv": 0.15},
    "manchester united": {"name": "Manchester United", "attack": 1.45, "defense": 1.40, "home_adv": 0.10},
    "real madrid": {"name": "Real Madrid", "attack": 2.40, "defense": 0.88, "home_adv": 0.32},
    "barcelona": {"name": "Barcelona", "attack": 2.05, "defense": 1.05, "home_adv": 0.28},
    "bayern munich": {"name": "Bayern Munich", "attack": 2.35, "defense": 1.10, "home_adv": 0.30},
    "inter milan": {"name": "Inter Milan", "attack": 1.95, "defense": 0.80, "home_adv": 0.25},
    "juventus": {"name": "Juventus", "attack": 1.60, "defense": 0.90, "home_adv": 0.18},
    "psg": {"name": "PSG", "attack": 2.15, "defense": 1.15, "home_adv": 0.20},
}
ALIASES = {"man city": "manchester city", "man utd": "manchester united", "man united": "manchester united", "real": "real madrid", "barca": "barcelona", "bayern": "bayern munich", "inter": "inter milan", "paris saint-germain": "psg"}
FORMATIONS = ["Not provided", "4-3-3", "4-2-3-1", "4-4-2", "3-5-2", "3-4-3", "4-1-4-1", "5-3-2", "Other"]
COMPETITIONS = ["Not specified", "UEFA Champions League", "UEFA Europa League", "UEFA Conference League", "Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1", "Domestic cup", "International / national team", "Other"]


def norm(name):
    key = " ".join(name.strip().lower().split())
    return ALIASES.get(key, key)


def poisson_pmf(k, lam):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def parse_form(text):
    """Parse recent scores entered as goals-for-goals-against, e.g. 2-1, 0-0, 1-3."""
    entries = []
    for token in text.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split("-")
        if len(parts) != 2:
            raise ValueError(f"Could not read form item '{token}'. Use scores like 2-1, 0-0, 1-3.")
        try:
            gf, ga = int(parts[0].strip()), int(parts[1].strip())
        except ValueError:
            raise ValueError(f"Could not read form item '{token}'. Use whole-number scores like 2-1.")
        if min(gf, ga) < 0 or max(gf, ga) > 20:
            raise ValueError("Form scores must be between 0 and 20 goals per team.")
        entries.append((gf, ga))
    return entries[-10:]


def form_adjustment(entries):
    """Conservative xG multiplier based on manually supplied recent results."""
    if not entries:
        return 1.0, None
    n = len(entries)
    gf_avg = sum(x for x, _ in entries) / n
    ga_avg = sum(y for _, y in entries) / n
    points = sum(3 if x > y else 1 if x == y else 0 for x, y in entries) / n
    # Use the result trend and scoring/conceding signal, but shrink strongly toward neutral.
    signal = (gf_avg - 1.35) * 0.10 - (ga_avg - 1.35) * 0.06 + (points - 1.35) * 0.045
    mult = max(0.82, min(1.18, 1.0 + signal))
    return mult, {"matches": n, "goals_for_avg": gf_avg, "goals_against_avg": ga_avg, "points_avg": points}


def context_multiplier(formation, motivation, absences, role_impact, form_entries, form_weight_pct):
    # Small transparent heuristics; these are not empirical player/formation models.
    attack = 1.0
    defense = 1.0
    formation_effects = {
        "4-3-3": (1.015, 1.005), "4-2-3-1": (1.01, 1.00), "4-4-2": (1.00, 1.00),
        "3-5-2": (1.005, 1.015), "3-4-3": (1.015, 0.99), "4-1-4-1": (0.995, 1.01),
        "5-3-2": (0.985, 1.025), "Other": (1.0, 1.0), "Not provided": (1.0, 1.0),
    }
    fa, fd = formation_effects.get(formation, (1.0, 1.0))
    attack *= fa
    defense *= fd
    motivation_effects = {"Not provided": 0.0, "Very low": -0.025, "Low": -0.012, "Normal": 0.0, "High": 0.012, "Very high": 0.02}
    mot = motivation_effects.get(motivation, 0.0)
    attack *= 1.0 + mot
    # Absence impact selected by user. Each absence has a capped, modest role-based effect.
    absence_penalty = min(0.18, max(0, absences) * role_impact * 0.025)
    attack *= 1.0 - absence_penalty
    defense *= 1.0 + min(0.12, max(0, absences) * role_impact * 0.018)
    form_mult, form_summary = form_adjustment(form_entries)
    # Slider controls how much manually supplied form signal is trusted, not the team's total strength.
    blended_form = 1.0 + (form_mult - 1.0) * (form_weight_pct / 100.0)
    attack *= blended_form
    return max(0.70, min(1.30, attack)), max(0.70, min(1.30, defense)), form_summary, absence_penalty


def calculate_matrix(home_xg, away_xg, max_goals=8, rho=-0.05):
    raw = {}
    for h in range(max_goals + 1):
        ph = poisson_pmf(h, home_xg)
        for a in range(max_goals + 1):
            pa = poisson_pmf(a, away_xg)
            tau = 1.0
            if h == 0 and a == 0:
                tau = 1 - home_xg * away_xg * rho
            elif h == 0 and a == 1:
                tau = 1 + home_xg * rho
            elif h == 1 and a == 0:
                tau = 1 + away_xg * rho
            elif h == 1 and a == 1:
                tau = 1 - rho
            raw[(h, a)] = max(0.0, ph * pa * max(0.0, tau))
    total = sum(raw.values())
    if total <= 0 or not math.isfinite(total):
        raise ValueError("Could not calculate a valid score distribution.")
    matrix = {score: p / total for score, p in raw.items()}
    home_win = sum(p for (h, a), p in matrix.items() if h > a)
    draw = sum(p for (h, a), p in matrix.items() if h == a)
    away_win = sum(p for (h, a), p in matrix.items() if h < a)
    return {
        "matrix": matrix, "home_win": home_win, "draw": draw, "away_win": away_win,
        "over_15": sum(p for (h, a), p in matrix.items() if h + a >= 2),
        "over_25": sum(p for (h, a), p in matrix.items() if h + a >= 3),
        "over_35": sum(p for (h, a), p in matrix.items() if h + a >= 4),
        "btts": sum(p for (h, a), p in matrix.items() if h > 0 and a > 0),
        "top_scores": sorted(matrix.items(), key=lambda item: item[1], reverse=True)[:10],
    }


def run_monte_carlo(home_xg, away_xg, n_sims, seed=42):
    rng = random.Random(seed)
    def draw_poisson(lam):
        # Knuth's algorithm, suitable for football-sized lambdas.
        if lam <= 0:
            return 0
        limit = math.exp(-lam)
        product = 1.0
        k = 0
        while product > limit:
            k += 1
            product *= rng.random()
        return k - 1
    counts = Counter()
    score_counts = Counter()
    total_goals = Counter()
    for _ in range(n_sims):
        h, a = draw_poisson(home_xg), draw_poisson(away_xg)
        if h > a:
            counts["home"] += 1
        elif h == a:
            counts["draw"] += 1
        else:
            counts["away"] += 1
        if h + a >= 2: counts["over15"] += 1
        if h + a >= 3: counts["over25"] += 1
        if h + a >= 4: counts["over35"] += 1
        if h > 0 and a > 0: counts["btts"] += 1
        score_counts[(h, a)] += 1
        total_goals[h + a] += 1
    return {
        "home": counts["home"] / n_sims, "draw": counts["draw"] / n_sims, "away": counts["away"] / n_sims,
        "over15": counts["over15"] / n_sims, "over25": counts["over25"] / n_sims,
        "over35": counts["over35"] / n_sims, "btts": counts["btts"] / n_sims,
        "top_scores": score_counts.most_common(10), "goal_distribution": total_goals,
    }


# ----------------------------- PAGE HEADER -----------------------------
st.markdown('<div class="main-title">⚽ MATCHLAB</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Context-aware football probability and simulation workspace · 2026/2027</div>', unsafe_allow_html=True)
st.markdown("""
<div class="welcome">
<h2>Welcome to MATCHLAB</h2>
<p>Input your data. Define your criteria. Run your simulation.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("⚽ Match Setup")
    st.caption("Choose a mode and enter the evidence you have.")
    simulation_mode = st.selectbox("Simulation mode", ["Standard Analysis", "Competition-Aware Analysis", "Lineup-Based Analysis", "Full Context Analysis", "Custom Simulation"])
    home_input = st.text_input("Home team", "Aston Villa")
    away_input = st.text_input("Away team", "Manchester City")
    competition = st.selectbox("Competition", COMPETITIONS, index=1 if simulation_mode != "Standard Analysis" else 0)
    season = st.selectbox("Season", ["2026/2027", "2025/2026", "Other / custom"])
    match_stage = st.selectbox("Match stage", ["Not specified", "League / league phase", "Group stage", "Knockout — first leg", "Knockout — second leg", "Final", "Domestic cup", "Friendly"])
    venue = st.selectbox("Venue", ["Home team venue", "Away team venue", "Neutral venue", "Not sure"])
    max_goals = st.selectbox("Scoreline limit per team", [5, 6, 7, 8, 9, 10, 12], index=3)
    st.divider()
    st.subheader("Custom simulation")
    form_weight = st.slider("Trust in manually entered form (%)", 5, 100, 70, 5)
    n_sims = st.select_slider("Monte Carlo simulations", options=[1000, 5000, 10000, 25000, 50000], value=10000)
    seed = st.number_input("Simulation seed (repeatable results)", min_value=1, max_value=999999, value=42, step=1)

if not home_input.strip() or not away_input.strip():
    st.error("Enter both team names to continue.")
    st.stop()
if norm(home_input) == norm(away_input):
    st.error("Home and away teams must be different.")
    st.stop()

st.subheader("1. Team performance evidence")
left, right = st.columns(2)
with left:
    st.markdown(f"#### {home_input.strip()}")
    home_form_text = st.text_input("Recent form — goals for-goals against", placeholder="Example: 2-1, 1-1, 0-2, 3-0, 2-2", key="home_form")
    home_form_pct = st.slider("Home form influence (%)", 5, 100, 70, 5)
    home_form_entries = []
    home_form_error = None
    try:
        home_form_entries = parse_form(home_form_text)
    except ValueError as e:
        home_form_error = str(e)
    if home_form_error: st.error(home_form_error)
    elif home_form_entries: st.caption(f"Read {len(home_form_entries)} recent results; most recent entries are used.")
with right:
    st.markdown(f"#### {away_input.strip()}")
    away_form_text = st.text_input("Recent form — goals for-goals against", placeholder="Example: 1-0, 0-1, 2-2, 2-1, 1-3", key="away_form")
    away_form_pct = st.slider("Away form influence (%)", 5, 100, 70, 5)
    away_form_entries = []
    away_form_error = None
    try:
        away_form_entries = parse_form(away_form_text)
    except ValueError as e:
        away_form_error = str(e)
    if away_form_error: st.error(away_form_error)
    elif away_form_entries: st.caption(f"Read {len(away_form_entries)} recent results; most recent entries are used.")

st.subheader("2. Optional lineup images")
img_col1, img_col2 = st.columns(2)
with img_col1:
    home_image = st.file_uploader(f"Upload {home_input.strip()} lineup image", type=["png", "jpg", "jpeg", "webp"], key="home_lineup")
    if home_image:
        st.image(home_image, caption="Home lineup image", use_container_width=True)
with img_col2:
    away_image = st.file_uploader(f"Upload {away_input.strip()} lineup image", type=["png", "jpg", "jpeg", "webp"], key="away_lineup")
    if away_image:
        st.image(away_image, caption="Away lineup image", use_container_width=True)
st.info("Lineup images are accepted and displayed for reference. This version does not automatically identify player names from images; type the extracted lineup below or use the image as a reference.")
lineup_a, lineup_b = st.columns(2)
with lineup_a:
    home_lineup_text = st.text_area(f"{home_input.strip()} lineup / notes (optional)", placeholder="Player names, confirmed XI, key absences, tactical notes…", height=90)
with lineup_b:
    away_lineup_text = st.text_area(f"{away_input.strip()} lineup / notes (optional)", placeholder="Player names, confirmed XI, key absences, tactical notes…", height=90)

st.subheader("3. Optional match context")
ctx1, ctx2 = st.columns(2)
with ctx1:
    st.markdown(f"#### {home_input.strip()} context")
    home_formation = st.selectbox("Formation", FORMATIONS, key="home_formation")
    home_motivation = st.selectbox("Motivation / objective", ["Not provided", "Very low", "Low", "Normal", "High", "Very high"], index=0, key="home_motivation")
    home_absences = st.number_input("Unavailable players", min_value=0, max_value=11, value=0, key="home_absences")
    home_role_impact = st.select_slider("Average importance of absentees", options=[0.5, 1.0, 1.5, 2.0], value=1.0, key="home_role_impact")
    home_absence_notes = st.text_input("Absence notes (optional)", placeholder="e.g. starting striker suspended", key="home_absence_notes")
    home_rest = st.slider("Rest since previous match (days)", 0, 14, 5, key="home_rest")
with ctx2:
    st.markdown(f"#### {away_input.strip()} context")
    away_formation = st.selectbox("Formation", FORMATIONS, key="away_formation")
    away_motivation = st.selectbox("Motivation / objective", ["Not provided", "Very low", "Low", "Normal", "High", "Very high"], index=0, key="away_motivation")
    away_absences = st.number_input("Unavailable players", min_value=0, max_value=11, value=0, key="away_absences")
    away_role_impact = st.select_slider("Average importance of absentees", options=[0.5, 1.0, 1.5, 2.0], value=1.0, key="away_role_impact")
    away_absence_notes = st.text_input("Absence notes (optional)", placeholder="e.g. first-choice goalkeeper injured", key="away_absence_notes")
    away_rest = st.slider("Rest since previous match (days)", 0, 14, 5, key="away_rest")

wx1, wx2 = st.columns(2)
with wx1:
    weather = st.selectbox("Match weather (optional)", ["Not provided", "Clear / mild", "Cloudy", "Light rain", "Heavy rain", "Strong wind", "Extreme heat", "Snow / icy pitch"])
with wx2:
    pitch = st.selectbox("Pitch condition (optional)", ["Not provided", "Good", "Wet", "Heavy / waterlogged", "Artificial", "Neutral venue"])

st.subheader("4. Run simulation")
st.caption("Review your inputs, then run the model. The result updates when you press the button.")
run = st.button("⚽ Run MATCHLAB simulation", type="primary", use_container_width=True)

if run:
    errors = [e for e in [home_form_error, away_form_error] if e]
    if errors:
        for e in errors: st.error(e)
        st.stop()

    home_data = TEAM_DATA.get(norm(home_input), {"name": home_input.strip(), "attack": 1.40, "defense": 1.40, "home_adv": 0.20})
    away_data = TEAM_DATA.get(norm(away_input), {"name": away_input.strip(), "attack": 1.40, "defense": 1.40, "home_adv": 0.20})
    home_attack_mult, home_def_mult, home_form_summary, home_absence_penalty = context_multiplier(home_formation, home_motivation, int(home_absences), float(home_role_impact), home_form_entries, int(form_weight * home_form_pct / 100))
    away_attack_mult, away_def_mult, away_form_summary, away_absence_penalty = context_multiplier(away_formation, away_motivation, int(away_absences), float(away_role_impact), away_form_entries, int(form_weight * away_form_pct / 100))

    # Competition name itself does not receive a blind bonus. Stage changes are modest and explicit.
    stage_attack_home = stage_attack_away = 1.0
    stage_notes = []
    if match_stage == "Knockout — second leg":
        stage_notes.append("Second-leg context noted; aggregate score is not supplied, so no automatic chasing/defending adjustment is applied.")
    elif match_stage == "Final":
        stage_notes.append("Final selected. No generic pressure penalty is applied without validated competition-specific evidence.")
    if venue == "Neutral venue":
        home_adv = 0.0
    elif venue == "Away team venue":
        home_adv = -0.12
    else:
        home_adv = float(home_data.get("home_adv", 0.20))

    # Rest disadvantage is deliberately modest and capped; not a live workload model.
    home_rest_mult = max(0.94, min(1.02, 1.0 + (home_rest - 5) * 0.006))
    away_rest_mult = max(0.94, min(1.02, 1.0 + (away_rest - 5) * 0.006))
    weather_factor = {"Not provided": 1.0, "Clear / mild": 1.0, "Cloudy": 1.0, "Light rain": 0.99, "Heavy rain": 0.96, "Strong wind": 0.97, "Extreme heat": 0.97, "Snow / icy pitch": 0.94}[weather]
    pitch_factor = {"Not provided": 1.0, "Good": 1.0, "Wet": 0.99, "Heavy / waterlogged": 0.96, "Artificial": 1.0, "Neutral venue": 1.0}[pitch]
    conditions_factor = max(0.90, weather_factor * pitch_factor)

    home_xg = home_data["attack"] * away_data["defense"] * home_attack_mult * away_def_mult * home_rest_mult * conditions_factor + home_adv
    away_xg = away_data["attack"] * home_data["defense"] * away_attack_mult * home_def_mult * away_rest_mult * conditions_factor
    home_xg = max(0.05, min(5.0, home_xg))
    away_xg = max(0.05, min(5.0, away_xg))

    exact = calculate_matrix(home_xg, away_xg, max_goals=max_goals)
    sims = run_monte_carlo(home_xg, away_xg, int(n_sims), int(seed))
    home_display, away_display = home_data["name"], away_data["name"]

    st.divider()
    st.header(f"{home_display} vs {away_display}")
    st.caption(f"Season: {season} · Competition: {competition} · Stage: {match_stage} · Mode: {simulation_mode}")
    if norm(home_input) not in TEAM_DATA or norm(away_input) not in TEAM_DATA:
        st.warning("At least one team uses generic default ratings. These are not team-specific or live 2026/2027 ratings.")
    st.warning("No external live-data provider is connected in this version. Competition, player notes and uploaded images are user-supplied context; lineup images are not automatically OCR-analysed. Treat outputs as exploratory estimates, not verified real-world forecasts.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Home expected goals", f"{home_xg:.2f}")
    m2.metric("Total expected goals", f"{home_xg + away_xg:.2f}")
    m3.metric("Away expected goals", f"{away_xg:.2f}")

    st.subheader("Match outcome probabilities")
    p1, p2, p3 = st.columns(3)
    for col, label, key in [(p1, f"{home_display} win", "home_win"), (p2, "Draw", "draw"), (p3, f"{away_display} win", "away_win")]:
        col.metric(label, f"{exact[key] * 100:.1f}%")
        col.caption(f"Monte Carlo: {sims[key.split('_')[0] if key != 'home_win' else 'home'] * 100:.1f}%" if key in ("home_win", "draw", "away_win") else "")

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Over 1.5 goals", f"{exact['over_15'] * 100:.1f}%")
    g2.metric("Over 2.5 goals", f"{exact['over_25'] * 100:.1f}%")
    g3.metric("Over 3.5 goals", f"{exact['over_35'] * 100:.1f}%")
    g4.metric("Both teams to score", f"{exact['btts'] * 100:.1f}%")

    best_outcome = max([(home_display, exact["home_win"]), ("Draw", exact["draw"]), (away_display, exact["away_win"])], key=lambda x: x[1])
    best_score, best_score_p = exact["top_scores"][0]
    st.subheader("Model summary")
    st.success(f"Most probable 1X2 outcome: {best_outcome[0]} ({best_outcome[1] * 100:.1f}%). This is the most likely of the three outcomes, not
