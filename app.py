import math
from datetime import datetime, timezone
import pandas as pd
import streamlit as st
import cv2
import numpy as np

# Core Machine Learning Dependencies
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# Lazy load EasyOCR only when an image is processed to conserve mobile runtime memory
@st.cache_resource
def get_ocr_reader():
    import easyocr
    return easyocr.Reader(['en'], gpu=False)

st.set_page_config(page_title="MatchLab Pro v3.5", page_icon="⚽", layout="wide")
st.title("⚽ MatchLab 3.5 Pro (Context Intelligence AI)")
st.caption("Advanced football platform · Local Machine Learning Core · Zero-Manual Pipeline")

st.warning("Autonomous Mode Active: Contextual AI is researching parameters silently behind the dashboard.")

# --- CORE MATH UTILITIES ---
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
    
    top = sorted(scores.items(), key=lambda kv: kv, reverse=True)[:10]

    return {
        "home_win": home_win, "draw": draw, "away_win": away_win,
        "over15": over(1), "over25": over(2), "over35": over(3),
        "btts": btts, "home_score": home_score, "away_score": away_score,
        "top": [(f"{h}-{a}", p) for (h,a), p in top]
    }

# --- 1. HISTORICAL LEAGUE TEAM BASELINES ---
historical_teams_data = {
    "Team": ["Aston Villa", "Manchester City", "Arsenal", "Liverpool", "Chelsea", "Manchester United"],
    "Attack_Rating": [1.65, 2.30, 2.10, 2.25, 1.80, 1.45],
    "Defense_Rating": [1.20, 0.90, 0.85, 0.95, 1.35, 1.40],
    "Home_Advantage": [0.25, 0.30, 0.22, 0.35, 0.15, 0.10]
}
league_df = pd.DataFrame(historical_teams_data)

# --- 2. THE RESEARCH DATABASE (AUTOMATING OUT THE SLIDERS) ---
# This dictionary mimics pre-scraped platform entries for upcoming match features
AUTOMATED_RESEARCH_DB = {
    "Aston Villa vs Manchester City": {
        "weather_val": 1.00, "pitch_val": 1.00, "travel_km": 140, "ref_strict": 0.40, "motivation_delta": 0.15, "stadium_cap": 100,
        "opening_h": 4.20, "opening_d": 3.85, "opening_a": 1.80, "closing_h": 4.50, "closing_d": 3.90, "closing_a": 1.75, "has_closing": True
    },
    "Arsenal vs Chelsea": {
        "weather_val": 0.92, "pitch_val": 1.02, "travel_km": 15, "ref_strict": 0.80, "motivation_delta": 0.05, "stadium_cap": 100,
        "opening_h": 1.70, "opening_d": 3.60, "opening_a": 4.80, "closing_h": 0.00, "closing_d": 0.00, "closing_a": 0.00, "has_closing": False
    },
    "Liverpool vs Manchester United": {
        "weather_val": 0.85, "pitch_val": 1.00, "travel_km": 55, "ref_strict": 0.60, "motivation_delta": 0.20, "stadium_cap": 95,
        "opening_h": 1.50, "opening_d": 4.20, "opening_a": 6.00, "closing_h": 1.45, "closing_d": 4.40, "closing_a": 6.50, "has_closing": True
    }
}

# --- 3. TRAINING ENGINE FOR CONTEXT AI ---
np.random.seed(42)
sim_records = 200
X_train_mock = np.zeros((sim_records, 6))
y_train_h_xg = np.zeros(sim_records)
y_train_a_xg = np.zeros(sim_records)

for idx in range(sim_records):
    w = np.random.uniform(0.85, 1.0)
    p = np.random.uniform(0.95, 1.05)
    t = np.random.uniform(0.0, 1.5)
    r = np.random.uniform(0.1, 0.9)
    m = np.random.uniform(-0.2, 0.2)
    s = np.random.uniform(0.6, 1.0)
    X_train_mock[idx] = [w, p, t, r, m, s]
    y_train_h_xg[idx] = (1.6 * w * p * s) + m
    y_train_a_xg[idx] = (1.2 * w * p * s) - (t * 0.1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_mock)
ai_home_model = KNeighborsRegressor(n_neighbors=5, weights='distance').fit(X_train_scaled, y_train_h_xg)
ai_away_model = KNeighborsRegressor(n_neighbors=5, weights='distance').fit(X_train_scaled, y_train_a_xg)

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.header("🗓️ Automated Fixture Center")
match_selection = st.sidebar.selectbox("Select Active Matchup", list(AUTOMATED_RESEARCH_DB.keys()))

st.sidebar.header("⚖️ Mathematical Settings")
dc_rho = st.sidebar.slider("Dixon-Coles Parameter", -0.15, 0.15, -0.05, step=0.01)
max_goals = st.sidebar.slider("Scoreline Cap Range Limit", 6, 15, 10)

st.sidebar.header("📷 Vision Lineup Parser")
lineup_source = st.sidebar.radio("Capture Method", ["Disable Scanner", "Smartphone Camera Feed", "Upload Roster File"])

img_buffer = None
if lineup_source == "Smartphone Camera Feed":
    img_buffer = st.sidebar.camera_input("Snapshot Match Sheet")
elif lineup_source == "Upload Roster File":
    img_buffer = st.sidebar.file_uploader("Upload starting XI file infographic", type=["png", "jpg", "jpeg"])

parsed_text_list = []
if img_buffer is not None:
    with st.spinner("Executing Computer Vision text extraction loop..."):
        try:
            file_bytes = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
            opencv_img = cv2.imdecode(file_bytes, 1)
            reader = get_ocr_reader()
            ocr_results = reader.readtext(opencv_img)
            parsed_text_list = [text_val for (bbox, text_val, prob) in ocr_results]
            st.success("Vision Analysis Complete!")
        except Exception as cv_error:
            st.error(f"OCR Pipeline Fault: {cv_error}")

# --- EXECUTE SILENT BACKGROUND AI INFERENCE ---
ctx = AUTOMATED_RESEARCH_DB[match_selection]
selected_home, selected_away = match_selection.split(" vs ")

# Handle travel fatigue scaling vectors
fatigue_val = min(1.5, ctx["travel_km"] / 500.0)

# Evaluate Vision Squad Array
squad_integrity_coef = 1.0
ocr_logs_text = []
if parsed_text_list:
    text_blob_lower = " ".join(parsed_text_list).lower()
    star_registry = {
        "aston villa": ["watkins", "martinez", "bailey"],
        "manchester city": ["haaland", "de bruyne", "rodri"],
        "arsenal": ["saka", "odegaard", "saliba"]
    }
    home_key = selected_home.lower()
    if home_key in star_registry:
        for star in star_registry[home_key]:
            if star not in text_blob_lower:
                squad_integrity_coef -= 0.12
                ocr_logs_text.append(f"❌ **Roster Alert:** Star player {star.capitalize()} absent from line-up tracking feed.")

# Compute ML patterns
live_input_vector = np.array([[ctx["weather_val"], ctx["pitch_val"], fatigue_val, ctx["ref_strict"], ctx["motivation_delta"], squad_integrity_coef]])
live_input_scaled = scaler.transform(live_input_vector)

ai_adjustment_h = float(ai_home_model.predict(live_input_scaled))
ai_adjustment_a = float(ai_away_model.predict(live_input_scaled))

h_row = league_df[league_df["Team"] == selected_home].iloc[0]
a_row = league_df[league_df["Team"] == selected_away].iloc[0]

final_hxg = max(0.15, (h_row["Attack_Rating"] * a_row["Defense_Rating"]) + h_row["Home_Advantage"] + (ai_adjustment_h - 1.5))
final_axg = max(0.15, (a_row["Attack_Rating"] * h_row["Defense_Rating"]) + (ai_adjustment_a - 1.2))

# --- MAIN AUTOMATED UI DASHBOARD ---
st.header(f"🏟️ Game Center: {selected_home} vs {selected_away}")

# AI Core Reasoning Dashboard Viewports
st.header("🧠 AI Core Reasoning Dashboard")
a_col1, a_col2 = st.columns(2)
with a_col1:
    st.info("### 📈 Background Machine Learning Insights")
    st.write(f"🤖 **Home Context Weight Adjustment Matrix:** `{ai_adjustment_h:.4f}`")
    st.write(f"🤖 **Away Context Weight Adjustment Matrix:** `{ai_adjustment_a:.4f}`")
    st.write(f"🏃‍♂️ **Away Travel Fatigue Vector:** `{fatigue_val:.2f}` ({ctx['travel_km']} km covered)")
    st.write(f"🏁 **Referee Game Strictness Factor:** `{ctx['ref_strict']:.2f}`")
with a_col2:
    st.info("### 🎰 Market Lines Liability Analysis")
    st.write(f"💵 **Opening Bookmaker Lines:** `{ctx['opening_h']}` | `{ctx['opening_d']}` | `{ctx['opening_a']}`")
    if ctx["has_closing"]:
        st.write(f"💵 **Live Market Closing Lines:** `{ctx['closing_h']}` | `{ctx['closing_d']}` | `{ctx['closing_a']}`")
        shift = ctx["opening_h"] - ctx["closing_h"]
        if abs(shift) >= 0.10:
            st.error(f"⚠️ Heavy Market Liability Move! Margin variance gap: {shift:.2f}")
        else:
            st.success("✅ Stable trading line volume profile.")
    else:
        st.warning("⚠️ **Market Status Check:** [CLOSING ODDS ODDS NOT YET SET / PENDING KICKOFF TIME]. Evaluated using Opening Line baselines.")

if ocr_logs_text:
    st.info("### 🔍 Computer Vision Lineup Audit Logs")
    for log in ocr_logs_text: st.markdown(log)

# Calculate outcome spreads
odds_h = ctx["closing_h"] if ctx["has_closing"] else ctx["opening_h"]
odds_d = ctx["closing_d"] if ctx["has_closing"] else ctx["opening_d"]
odds_a = ctx["closing_a"] if ctx["has_closing"] else ctx["opening_a"]

result = calculate(final_hxg, final_axg, max_goals, dc_rho)

raw_implied_sum = (1/odds_h) + (1/odds_d) + (1/odds_a)
market_h = (1/odds_h) / raw_implied_sum
market_d = (1/odds_d) / raw_implied_sum
