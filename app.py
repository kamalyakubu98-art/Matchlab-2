import math
from datetime import datetime, timezone
import pandas as pd
import streamlit as st
import cv2
import numpy as np
import requests

# Lazy load EasyOCR only when an image is processed to conserve mobile runtime memory
@st.cache_resource
def get_ocr_reader():
    import easyocr
    return easyocr.Reader(['en'], gpu=False)

st.set_page_config(page_title="MatchLab Pro v3.5", page_icon="⚽", layout="wide")
st.title("⚽ MatchLab 3.5 Pro (Context Intelligence)")
st.caption("Advanced football probability platform · Live API Automation · Explanatory Context Logic")

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
    
    # FIXED MATRIX SORT LINE
    top = sorted(scores.items(), key=lambda kv: kv, reverse=True)[:10]

    return {
        "home_win": home_win, "draw": draw, "away_win": away_win,
        "over15": over(1), "over25": over(2), "over35": over(3),
        "btts": btts, "home_score": home_score, "away_score": away_score,
        "top": [(f"{h}-{a}", p) for (h,a), p in top]
    }

# --- SIDEBAR: Live Data Automation Framework ---
st.sidebar.header("🔑 1. API-Sports Connection")
api_key = st.sidebar.text_input("Enter API-Sports Key", type="password")
league_id = st.sidebar.number_input("League ID (EPL = 39)", min_value=1, value=39)

home_team = "Aston Villa"
away_team = "Manchester City"
odds_h, odds_d, odds_a = 2.10, 3.40, 3.20
base_hxg, base_axg = 1.55, 1.10

# Initialize live request pipelines if token exists
if api_key:
    headers = {'x-apisports-key': api_key}
    
    # Fetch upcoming fixture cards
    with st.sidebar.spinner("Fetching live fixture schedule..."):
        try:
            url = f"https://api-sports.io{league_id}&next=10"
            res = requests.get(url, headers=headers).json()
            
            if "response" in res and res["response"]:
                fixture_options = {}
                for f in res["response"]:
                    h_name = f['teams']['home']['name']
                    a_name = f['teams']['away']['name']
                    label = f"{h_name} vs {a_name}"
                    fixture_options[label] = f
                    
                selected_match_str = st.sidebar.selectbox("Select Active Live Fixture", list(fixture_options.keys()))
                
                if selected_match_str:
                    match_data = fixture_options[selected_match_str]
                    fixture_id = match_data["fixture"]["id"]
                    home_team = match_data["teams"]["home"]["name"]
                    away_team = match_data["teams"]["away"]["name"]
                    
                    # Call API Odds data layer for selected fixture match ID
                    odds_url = f"https://api-sports.io{fixture_id}"
                    odds_res = requests.get(odds_url, headers=headers).json()
                    
                    if "response" in odds_res and odds_res["response"]:
                        data_block = odds_res["response"][0]
                        bookmakers = data_block.get("bookmakers", [])
                        if bookmakers:
                            markets = bookmakers[0].get("markets", [])
                            for m in markets:
                                if m["name"] == "Match Winner":
                                    for o in m["outcomes"]:
                                        if o["name"] == "Home":
                                            odds_h = float(o["price"])
                                        elif o["name"] == "Draw":
                                            odds_d = float(o["price"])
                                        elif o["name"] == "Away":
                                            odds_a = float(o["price"])
            else:
                st.sidebar.error("No upcoming matches found or invalid token.")
        except Exception as api_err:
            st.sidebar.error(f"API Fetch Failure: {api_err}")

st.sidebar.header("📊 2. Match Settings Override")
if not api_key:
    home_team = st.sidebar.text_input("Home team", "Aston Villa")
    away_team = st.sidebar.text_input("Away team", "Manchester City")

dc_rho = st.sidebar.slider("Dixon-Coles Adjustment Factor", -0.15, 0.15, -0.05, step=0.01)
max_goals = st.sidebar.slider("Maximum scoreline cap", 6, 15, 10)

st.sidebar.header("📷 3. Visual Lineup Input")
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

st.sidebar.header("📈 4. Market Context Lines")
odds_h = st.sidebar.number_input("Home Win Odds", min_value=1.01, value=odds_h)
odds_d = st.sidebar.number_input("Draw Odds", min_value=1.01, value=odds_d)
odds_a = st.sidebar.number_input("Away Win Odds", min_value=1.01, value=odds_a)

# --- MAIN UI LAYOUT ---
st.header("🎛️ Environmental & Situational Modifiers")

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

weather_multiplier = 1.0 if "Perfect" in weather_factor else (0.92 if "Heavy Rain" in weather_factor else 0.85)
pitch_multiplier = 1.0 if "Immaculate" in pitch_factor else (0.95 if "Slow" in pitch_factor else 1.04)
stadium_multiplier = stadium_cap / 100.0

# --- SQUAD REGISTER DATABASE ---
roster_db = {
    "aston villa": [
        {"name": "Watkins", "pos": "FW", "weight": 0.25, "reason": "Responsible for 31% of team open-play transitions"},
        {"name": "Martinez", "pos": "GK", "weight": 0.15, "reason": "Primary shot-stopping variance anchor"}
    ],
    "manchester city": [
        {"name": "Haaland", "pos": "FW", "weight": 0.35, "reason": "Core box target responsible for 42% of conversion metrics"},
        {"name": "De Bruyne", "pos": "MF", "weight": 0.20, "reason": "Main creativity engine supplying offensive production"}
    ],
    "arsenal": [
        {"name": "Saka", "pos": "FW", "weight": 0.20, "reason": "Primary creative driver and direct box entry point"},
        {"name": "Odegaard", "pos": "MF", "weight": 0.15, "reason": "Central transition link and defensive pressing director"}
    ]
}

xG_downgrade_reasons = []
lineup_penalty_multiplier_h = 1.0
lineup_penalty_multiplier_a = 1.0

if parsed_text_list:
    text_blob_lower = " ".join(parsed_text_list).lower()
    
    home_key = home_team.strip().lower()
    if home_key in roster_db:
        for player in roster_db[home_key]:
            if player["name"].lower() not in text_blob_lower:
                lineup_penalty_multiplier_h -= player["weight"]
                xG_downgrade_reasons.append(
                    f"⚠️ **{home_team} Downside:** {player['name']} ({player['pos']}) missing from analyzed lineup. "
                    f"Baseline dropped by -{player['weight']*100:.0f}%. Reason: {player['reason']}."
