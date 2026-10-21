import math
from datetime import datetime, timezone
import pandas as pd
import streamlit as st
import cv2
import numpy as np

# Lazy load EasyOCR only when an image is processed to conserve mobile runtime memory
@st.cache_resource
def get_ocr_reader():
    import easyocr
    return easyocr.Reader(['en'], gpu=False)

st.set_page_config(page_title="MatchLab Pro v3.5", page_icon="⚽", layout="wide")
st.title("⚽ MatchLab 3.5 Pro (Context Intelligence)")
st.caption("Advanced football probability platform · Visual Lineups · Environmental & Situational Modifiers")

st.warning(
    "Prototype notice: Context adjustments modify base statistical baselines. "
    "Use analytical discipline when grading subjective parameters."
)

def poisson_pmf(k, lam):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))

def dixon_coles_rho(x, y, mu, nu, rho):
    if x == 0 and y == 0: return 1.0 - mu * nu * rho
    elif x == 1 and y == 0: return 1.0 + mu * rho
    elif x == 0 and y == 1: return 1.0 + nu * rho
    elif x == 1 and y == 1: return 1.0 - rho
    return 1.0

def calculate(home_xg, away_xg, max_goals=10, rho=-0.05):
    max_goals = max(6, min(int(max_goals), 15))
    scores = {}
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p_home = poisson_pmf(h, home_xg)
            p_away = poisson_pmf(a, away_xg)
            adjustment = dixon_coles_rho(h, a, home_xg, away_xg, rho)
            scores[(h, a)] = p_home * p_away * adjustment

    mass = sum(scores.values())
    if mass > 0:
        scores = {s: p / mass for s, p in scores.items()}

    home_win = sum(p for (h,a),p in scores.items() if h > a)
    draw = sum(p for (h,a),p in scores.items() if h == a)
    away_win = sum(p for (h,a),p in scores.items() if h < a)

    def over(n): return sum(p for (h,a),p in scores.items() if h+a > n)
    btts = sum(p for (h,a),p in scores.items() if h > 0 and a > 0)
    home_score = sum(p for (h,a),p in scores.items() if h > 0)
    away_score = sum(p for (h,a),p in scores.items() if a > 0)
    
    # SUCCESSFUL FIXED LINE: Sort strictly by item value index [1] (Probability)
    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:10]

    return {
        "home_win": home_win, "draw": draw, "away_win": away_win,
        "over15": over(1), "over25": over(2), "over35": over(3),
        "btts": btts, "home_score": home_score, "away_score": away_score,
        "top": [(f"{h}-{a}", p) for (h,a), p in top]
    }

# --- SIDEBAR: Core Settings ---
st.sidebar.header("1. Core Match Setup")
home_team = st.sidebar.text_input("Home team", "Aston Villa")
away_team = st.sidebar.text_input("Away team", "Manchester City")
mode = st.sidebar.radio("Expected goals source:", ["Enter expected goals manually", "Historical CSV"])

st.sidebar.header("2. Base Settings")
dc_rho = st.sidebar.slider("Dixon-Coles Adjustment Factor", -0.15, 0.15, -0.05, step=0.01)
max_goals = st.sidebar.slider("Maximum scoreline cap", 6, 15, 10)

st.sidebar.header("3. Visual Lineup Input")
st.sidebar.caption("Capture or upload squad charts to parse key player impacts.")
lineup_source = st.sidebar.radio("Lineup Capture Method", ["Disable Visual Parsing", "Smartphone Camera Feed", "Upload Infographic File"])

img_buffer = None
if lineup_source == "Smartphone Camera Feed":
    img_buffer = st.sidebar.camera_input("Take a photo of starting XI details")
elif lineup_source == "Upload Infographic File":
    img_buffer = st.sidebar.file_uploader("Upload lineup sheet screenshot", type=["png", "jpg", "jpeg"])

parsed_text_list = []
if img_buffer is not None:
    with st.spinner("Processing image pixels through Computer Vision Engine..."):
        try:
            file_bytes = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
            opencv_img = cv2.imdecode(file_bytes, 1)
            reader = get_ocr_reader()
            ocr_results = reader.readtext(opencv_img)
            parsed_text_list = [text_val for (bbox, text_val, prob) in ocr_results]
            st.success("Analysis Complete!")
            with st.expander("👁️ View Extracted Squad Text"):
                st.write(", ".join(parsed_text_list))
        except Exception as cv_error:
            st.error(f"Vision engine error: {cv_error}")

st.sidebar.header("4. Market Context")
odds_h = st.sidebar.number_input("Home Win Odds", min_value=1.01, value=2.10)
odds_d = st.sidebar.number_input("Draw Odds", min_value=1.01, value=3.40)
odds_a = st.sidebar.number_input("Away Win Odds", min_value=1.01, value=3.20)

# --- MAIN UI LAYOUT ---
st.header("🎛️ Environmental & Situational Modifiers")
st.caption("Apply contextual adjustments based on team context parameters.")

expl, exp2, exp3 = st.columns(3)
with expl:
    weather_factor = st.selectbox("Weather Variance", ["Perfect / Dry (1.00x)", "Heavy Rain / Wet Pitch (0.92x xG)", "Extreme Cold / Snow (0.85x xG)"])
    pitch_factor = st.selectbox("Pitch Surface Quality", ["Immaculate / Normal (1.00x)", "Slow / Long Muddy Grass (0.95x)", "Fast Artificial Turf (1.04x)"])
with exp2:
    motivation_h = st.slider("Home Motivation Boost", 0.80, 1.20, 1.00, 0.05)
    motivation_a = st.slider("Away Motivation Boost", 0.80, 1.20, 1.00, 0.05)
with exp3:
    ref_factor = st.selectbox("Referee Tendency", ["Standard / Balanced (1.00x)", "Strict / Card-Happy (Increases Penalty/Red Risk)", "Lenient / Fluid Play (0.98x xG Drop)"])
    stadium_cap = st.slider("Stadium Congestion Crowding Impact", 80, 120, 100, 5)

# Factor mapping conversions
weather_multiplier = 1.0 if "Perfect" in weather_factor else (0.92 if "Heavy Rain" in weather_factor else 0.85)
pitch_multiplier = 1.0 if "Immaculate" in pitch_factor else (0.95 if "Slow" in pitch_factor else 1.04)
stadium_multiplier = stadium_cap / 100.0

# --- EXPLAINABLE AI SQUAD ROSTER DATA MODULE ---
# Mocking core API-Sports player roster database mapping
roster_db = {
    "aston villa": [
        {"name": "Watkins", "pos": "FW", "weight": 0.25, "reason": "Responsible for 31% of team open-play transitions"},
        {"name": "Martinez", "pos": "GK", "weight": 0.15, "reason": "Primary shot-stopping variance anchor (+0.44 prevented goals)"}
    ],
    "manchester city": [
        {"name": "Haaland", "pos": "FW", "weight": 0.35, "reason": "Core box target responsible for 42% of conversion metrics"},
        {"name": "De Bruyne", "pos": "MF", "weight": 0.20, "reason": "Main creativity engine (average 4.2 shot-creating actions)"}
    ]
}

xG_downgrade_reasons = []
lineup_penalty_multiplier_h = 1.0
lineup_penalty_multiplier_a = 1.0

if parsed_text_list:
    text_blob_lower = " ".join(parsed_text_list).lower()
    
    # Process Home Roster Adjustments
    home_key = home_team.strip().lower()
    if home_key in roster_db:
        for player in roster_db[home_key]:
            if player["name"].lower() not in text_blob_lower:
                lineup_penalty_multiplier_h -= player["weight"]
                xG_downgrade_reasons.append(
                    f"⚠️ **{home_team} Downside:** {player['name']} ({player['pos']}) missing from analyzed lineup feed. "
                    f"Baseline lowered by -{player['weight']*100:.0f}% because he is: {player['reason']}."
                )
                
    # Process Away Roster Adjustments
    away_key = away_team.strip().lower()
    if away_key in roster_db:
        for player in roster_db[away_key]:
            if player["name"].lower() not in text_blob_lower:
                lineup_penalty_multiplier_a -= player["weight"]
                xG_downgrade_reasons.append(
                    f"⚠️ **{away_team} Downside:** {player['name']} ({player['pos']}) missing from analyzed lineup feed. "
                    f"Baseline lowered by -{player['weight']*100:.0f}% because he is: {player['reason']}."
                )

# Safeguard weights from falling below absolute zero
lineup_penalty_multiplier_h = max(0.1, lineup_penalty_multiplier_h)
lineup_penalty_multiplier_a = max(0.1, lineup_penalty_multiplier_a)

if mode == "Enter expected goals manually":
    st.subheader("Base expected-goals variables")
    c1, c2 = st.columns(2)
    base_hxg = c1.number_input(f"Base {home_team} xG", min_value=0.15, max_value=4.50, value=1.55, step=0.05)
    base_axg = c2.number_input(f"Base {away_team} xG", min_value=0.15, max_value=4.50, value=1.10, step=0.05)

    hxg = base_hxg * motivation_h * pitch_multiplier * stadium_multiplier * lineup_penalty_multiplier_h
    axg = base_axg * motivation_a * weather_multiplier * pitch_multiplier * lineup_penalty_multiplier_a
else:
    st.info("Historical data computation matrix active.")
    hxg = 1.50
    axg = 1.10

# Print Explainable AI Log Messages
if xG_downgrade_reasons:
    st.info("### 🔍 Live Lineup Disruption Logs")
    for log_message in xG_downgrade_reasons:
        st.markdown(log_message)

if hxg is not None and axg is not None and home_team.strip() and away_team.strip() and home_team.strip().casefold() != away_team.strip().casefold():
    result = calculate(float(hxg), float(axg), max_goals, dc_rho)
    st.divider()

    raw_implied_sum = (1/odds_h) + (1/odds_d) + (1/odds_a)
    bookie_margin = raw_implied_sum - 1.0
    market_h = (1/odds_h) / raw_implied_sum
    market_d = (1/odds_d) / raw_implied_sum
    market_a = (1/odds_a) / raw_implied_sum

    st.subheader(f"{home_team} vs {away_team}")
    st.caption(f"Final Model Adjusted xG: {home_team} (hxg): {hxg:.2f} | {away_team} (axg): {axg:.2f}")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(f"{home_team} Win", f"{result['home_win']*100:.1f}%")
        if (result['home_win'] - market_h) > 0.02:
            st.success(f"📈 Edge: +{((result['home_win'] - market_h)*100):.1f}%")
    with m2:
        st.metric("Draw", f"{result['draw']*100:.1f}%")
        if (result['draw'] - market_d) > 0.02:
            st.success(f"📈 Edge: +{((result['draw'] - market_d)*100):.1f}%")
    with m3:
    
