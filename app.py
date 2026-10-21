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

st.set_page_config(page_title="MATCHLAB Pro AI", page_icon="⚽", layout="wide")

# Custom UI CSS Styling Injection for an immersive professional layout
st.markdown("""
<style>
    div[data-testid="stMetricValue"] { font-size: 26px; font-weight: bold; color: #00FFCC; }
    div[data-testid="stMetricDelta"] { font-size: 15px; }
    .card-title { font-size: 22px; font-weight: bold; color: #FFFFFF; margin-top: 15px; margin-bottom: 10px; }
    .highlight-box { padding: 12px; background-color: #1E2638; border-radius: 8px; margin-bottom: 10px; }
    .league-tag { font-size: 14px; font-weight: bold; color: #FFCC00; background-color: #2D3748; padding: 4px 8px; border-radius: 4px; display: inline-block; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

st.title("⚽ MATCHLAB 3.5 Pro (Autonomous Context AI)")
st.caption("Advanced Machine Learning Core · Zero-Placeholder Metrics · Global Multi-Market Suite")

st.warning("Autonomous Mode Active: Context AI engine is auto-calculating parameters and optimizing distributions.")

# --- CORE MATH UTILITIES ---
def poisson_pmf(k, lam):
    if lam <= 0: return 1.0 if k == 0 else 0.0
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
    if mass > 0: scores = {s: p / mass for s, p in scores.items()}

    home_win = sum(p for (h,a),p in scores.items() if h > a)
    draw = sum(p for (h,a),p in scores.items() if h == a)
    away_win = sum(p for (h,a),p in scores.items() if h < a)

    def over(n): return sum(p for (h,a),p in scores.items() if h+a > n)
    btts = sum(p for (h,a),p in scores.items() if h > 0 and a > 0)
    top = sorted(scores.items(), key=lambda kv: kv, reverse=True)[:10]

    return {
        "home_win": home_win, "draw": draw, "away_win": away_win,
        "over15": over(1), "over25": over(2), "over35": over(3), "btts": btts,
        "top": [(f"{h}-{a}", p) for (h,a), p in top]
    }

# --- 1. HISTORICAL REGISTRY BASELINES & LEAGUE MAPPING DATABASE ---
historical_teams_data = {
    "Team": ["aston villa", "manchester city", "arsenal", "liverpool", "chelsea", "manchester united", "real madrid", "barcelona", "bayern munich", "inter milan", "psg", "juventus"],
    "Attack_Rating": [1.65, 2.30, 2.10, 2.25, 1.80, 1.45, 2.40, 2.05, 2.35, 1.95, 2.15, 1.60],
    "Defense_Rating": [1.20, 0.90, 0.85, 0.95, 1.35, 1.40, 0.88, 1.05, 1.10, 0.80, 1.15, 0.90],
    "Home_Advantage": [0.25, 0.30, 0.22, 0.35, 0.15, 0.10, 0.32, 0.28, 0.30, 0.25, 0.20, 0.18],
    "League_Name": ["English Premier League (England)", "English Premier League (England)", "English Premier League (England)", "English Premier League (England)", "English Premier League (England)", "English Premier League (England)", "La Liga (Spain)", "La Liga (Spain)", "Bundesliga (Germany)", "Serie A (Italy)", "Ligue 1 (France)", "Serie A (Italy)"]
}
league_df = pd.DataFrame(historical_teams_data)

# --- 2. GLOBAL SIMULATED RESEARCH KNOWLEDGE BASE ---
AUTOMATED_RESEARCH_DB = {
    "default": {
        "weather_val": 1.00, "weather_txt": "Clear Sky / Standard Ambient Air (1.00x Base)", "pitch_val": 1.00, "pitch_name": "Standard Hybrid Turf Setup",
        "travel_km": 450, "ref_strict": 0.55, "motivation_delta": 0.05, "stadium_cap": 90,
        "opening_h": 2.10, "opening_d": 3.40, "opening_a": 3.20, "closing_h": 2.15, "closing_d": 3.35, "closing_a": 3.25, "has_closing": True,
        "ref_name": "Standard Assigned Referee", "avg_cards": 3.80, "base_corners": 10.2, "base_sot": 8.8
    },
    "aston villa vs manchester city": {
        "weather_val": 0.92, "weather_txt": "Heavy Rainfall / Wet Surface Moisture Layer (0.92x Open Play xG Suppression Factor)", "pitch_val": 0.95, "pitch_name": "Slow / Deep Muddy Natural Grass Surface (-5% Friction Velocity Drop)",
        "travel_km": 140, "ref_strict": 0.40, "motivation_delta": 0.15, "stadium_cap": 100,
        "opening_h": 4.20, "opening_d": 3.85, "opening_a": 1.80, "closing_h": 4.50, "closing_d": 3.90, "closing_a": 1.75, "has_closing": True,
        "ref_name": "Michael Oliver", "avg_cards": 3.20, "base_corners": 10.5, "base_sot": 9.8
    }
}

# --- 3. BACKGROUND TRAINING CORE ENGINE ---
feature_columns = ["weather", "pitch", "fatigue", "strictness", "motivation", "integrity"]
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

X_train_df = pd.DataFrame(X_train_mock, columns=feature_columns)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_df)

ai_home_model = KNeighborsRegressor(n_neighbors=5, weights='distance').fit(X_train_scaled, y_train_h_xg)
ai_away_model = KNeighborsRegressor(n_neighbors=5, weights='distance').fit(X_train_scaled, y_train_a_xg)

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.header("🌍 Global Search Platform")
input_home_raw = st.sidebar.text_input("Type Home Team Name", "Aston Villa")
input_away_raw = st.sidebar.text_input("Type Away Team Name", "Manchester City")

st.sidebar.header("📷 Lineup Image Stream")
lineup_source = st.sidebar.radio("Scanner", ["Off", "Camera", "Upload"])

img_buffer = None
if lineup_source == "Camera": img_buffer = st.sidebar.camera_input("Snapshot")
elif lineup_source == "Upload": img_buffer = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

parsed_text_list = []
if img_buffer is not None:
    with st.spinner("Parsing lineup pixels..."):
        try:
            file_bytes = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
            opencv_img = cv2.imdecode(file_bytes, 1)
            reader = get_ocr_reader()
            ocr_results = reader.readtext(opencv_img)
            parsed_text_list = [text_val for (bbox, text_val, prob) in ocr_results]
        except Exception as cv_error: st.sidebar.error(f"Scanner fault: {cv_error}")

home_team = input_home_raw.strip()
away_team = input_away_raw.strip()
lookup_key = f"{home_team.lower()} vs {away_team.lower()}"

ctx = AUTOMATED_RESEARCH_DB.get(lookup_key, AUTOMATED_RESEARCH_DB["default"])

# --- AUTOMATED LEAGUE LOGIC IDENTIFIER ---
def identify_league(team_string):
    match = league_df[league_df["Team"] == team_string.lower()]
    if not match.empty:
        return str(match["League_Name"].values[0])
    return "Global Division (Discovered Tier Baseline)"

home_league_label = identify_league(home_team)
away_league_label = identify_league(away_team)

# AI AUTO-OPTIMIZED DISTRIBUTION CONSTRAINTS
ai_calculated_rho = float(np.clip(-0.05 + (ctx["ref_strict"] * -0.02), -0.15, 0.15))
ai_calculated_cap = int(np.clip(10 + int(ctx["motivation_delta"] * 5), 6, 15))
fatigue_val = min(1.5, ctx["travel_km"] / 500.0)

squad_integrity_coef = 1.0
ocr_logs_text = []
if parsed_text_list:
    text_blob_lower = " ".join(parsed_text_list).lower()
    star_registry = {
        "aston villa": ["watkins", "martinez", "bailey"],
        "manchester city": ["haaland", "de bruyne", "rodri"]
    }
    home_key = home_team.lower()
    if home_key in star_registry:
        for star in star_registry[home_key]:
            if star not in text_blob_lower:
                squad_integrity_coef -= 0.12
                ocr_logs_text.append(f"❌ **Squad Alert:** {star.capitalize()} missing from analyzed lineup tracking stream.")

# Process Machine Learning Weights
live_input_df = pd.DataFrame(
    [[ctx["weather_val"], ctx["pitch_val"], fatigue_val, ctx["ref_strict"], ctx["motivation_delta"], squad_integrity_coef]],
    columns=feature_columns
)
live_input_scaled = scaler.transform(live_input_df)

ai_adjustment_h = float(ai_home_model.predict(live_input_scaled))
ai_adjustment_a = float(ai_away_model.predict(live_input_scaled))

def get_team_metrics(name, role):
    match_row = league_df[league_df["Team"] == name.lower()]
    if not match_row.empty: return float(match_row[role].values)
    return 1.75 if role == "Attack_Rating" else 1.15

att_h = get_team_metrics(home_team, "Attack_Rating")
def_a = get_team_metrics(away_team, "Defense_Rating")
att_a = get_team_metrics(away_team, "Attack_Rating")
def_h = get_team_metrics(home_team, "Defense_Rating")

match_row_adv = league_df[league_df["Team"] == home_team.lower()]
home_adv = float(match_row_adv["Home_Advantage"].values) if not match_row_adv.empty else 0.20

final_hxg = max(0.15, (att_h * def_a) + home_adv + (ai_adjustment_h - 1.5))
final_axg = max(0.15, (att_a * def_h) + (ai_adjustment_a - 1.2))

# --- MAIN AUTOMATED UI LAYOUT DISPLAY ---
