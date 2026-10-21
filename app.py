import math
from datetime import datetime, timezone
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MatchLab 2.0 Pro", page_icon="⚽", layout="wide")
st.title("⚽ MatchLab 2.0 Pro")
st.caption("Advanced football probability lab • Dixon-Coles adjustment + Time-decay form weighting")

st.warning(
    "Prototype notice: this app outputs statistical models based on historical data. "
    "Probabilities are mathematical estimates, not guaranteed outcomes."
)

def poisson_pmf(k, lam):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))

def dixon_coles_rho(x, y, mu, nu, rho):
    """Applies the Dixon-Coles adjustment factor for low scoring interdependence."""
    if x == 0 and y == 0:
        return 1.0 - mu * nu * rho
    elif x == 1 and y == 0:
        return 1.0 + nu * rho
    elif x == 0 and y == 1:
        return 1.0 + mu * rho
    elif x == 1 and y == 1:
        return 1.0 - rho
    return 1.0

def calculate(home_xg, away_xg, max_goals=10, rho=-0.05):
    """Calculate match probabilities using Dixon-Coles adjusted Poisson distribution."""
    max_goals = max(6, min(int(max_goals), 15))
    
    scores = {}
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p_home = poisson_pmf(h, home_xg)
            p_away = poisson_pmf(a, away_xg)
            adjustment = dixon_coles_rho(h, a, home_xg, away_xg, rho)
            scores[(h, a)] = p_home * p_away * adjustment
            
    # Normalize probabilities to ensure they sum to exactly 1.0
    mass = sum(scores.values())
    if mass > 0:
        scores = {s: p / mass for s, p in scores.items()}
    
    home_win = sum(p for (h,a),p in scores.items() if h > a)
    draw = sum(p for (h,a),p in scores.items() if h == a)
    away_win = sum(p for (h,a),p in scores.items() if h < a)
    
    def over(n):
        return sum(p for (h,a),p in scores.items() if h+a > n)
        
    btts = sum(p for (h,a),p in scores.items() if h > 0 and a > 0)
    home_score = sum(p for (h,a),p in scores.items() if h > 0)
    away_score = sum(p for (h,a),p in scores.items() if a > 0)
    
    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:10]
    
    return {
        "home_win": home_win, "draw": draw, "away_win": away_win,
        "over15": over(1), "over25": over(2), "over35": over(3),
        "btts": btts, "home_score": home_score, "away_score": away_score,
        "top": [(f"{h}-{a}", p) for (h,a),p in top],
        "grid_mass": mass
    }

def estimate_xg_weighted(df, home_name, away_name, lookback=10, decay=0.005):
    """Estimate expected goals using time-decay weighting (recent matches matter more)."""
    d = df.copy()
    d["home_team"] = d["home_team"].astype(str).str.strip()
    d["away_team"] = d["away_team"].astype(str).str.strip()
    
    for col in ["home_goals", "away_goals"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["home_team", "away_team", "home_goals", "away_goals"])
    d = d[(d.home_goals >= 0) & (d.away_goals >= 0)]
    
    if len(d) < 10:
        raise ValueError("Please upload at least 10 completed matches for a basic estimate.")
        
    # Apply time weights based on row recency if dates are missing, or by days elapsed if present
    if "date" in d.columns:
        d["_date"] = pd.to_datetime(d["date"], errors="coerce", utc=True)
        d = d.sort_values("_date").dropna(subset=["_date"])
        max_date = d["_date"].max()
        days_elapsed = (max_date - d["_date"]).dt.days
        d["weight"] = math.e ** (-decay * days_elapsed)
    else:
        # Fallback to row order index weighting if no dates exist
        d = d.reset_index(drop=True)
        d["weight"] = math.e ** (-decay * (len(d) - d.index))

    home_matches = d[d["home_team"].str.casefold() == home_name.strip().casefold()].tail(lookback)
    away_matches = d[d["away_team"].str.casefold() == away_name.strip().casefold()].tail(lookback)
    
    if home_matches.empty:
        raise ValueError(f"No home matches found for '{home_name}' in the uploaded file.")
    if away_matches.empty:
        raise ValueError(f"No away matches found for '{away_name}' in the uploaded file.")
        
    # Calculate weighted averages instead of flat means
    def weighted_mean(series, weights):
        return float((series * weights).sum() / weights.sum()) if weights.sum() > 0 else float(series.mean())

    league_home = weighted_mean(d["home_goals"], d["weight"])
    league_away = weighted_mean(d["away_goals"], d["weight"])
    
    h_scored = weighted_mean(home_matches["home_goals"], home_matches["weight"])
    h_conceded = weighted_mean(home_matches["away_goals"], home_matches["weight"])
    a_scored = weighted_mean(away_matches["away_goals"], away_matches["weight"])
    a_conceded = weighted_mean(away_matches["home_goals"], away_matches["weight"])
    
    # Calculate Attack and Defense Strengths relative to weighted league averages
    home_attack_strength = h_scored / max(league_home, 0.1)
    away_defense_strength = a_conceded / max(league_home, 0.1)
    away_attack_strength = a_scored / max(league_away, 0.1)
    home_defense_strength = h_conceded / max(league_away, 0.1)
    
    home_xg = home_attack_strength * away_defense_strength * league_home
    away_xg = away_attack_strength * home_defense_strength * league_away
    
    return max(0.15, min(home_xg, 4.5)), max(0.15, min(away_xg, 4.5)), {
        "matches_in_file": len(d), 
        "home_venue_sample": len(home_matches),
        "away_venue_sample": len(away_matches), 
        "weighted_league_home": league_home,
        "weighted_league_away": league_away
    }

with st.sidebar:
    st.header("1. Select match")
    home_team = st.text_input("Home team", "Manchester City")
    away_team = st.text_input("Away team", "Arsenal")
    mode = st.radio("How should expected goals be estimated?", ["Historical CSV", "Enter expected goals manually"])
    
    st.header("2. Model settings")
    lookback = st.slider("Recent venue matches to use", 5, 20, 10) if mode == "Historical CSV" else 10
    time_decay = st.slider("Time decay rate (Form sensitivity)", 0.000, 0.020, 0.005, step=0.001, help="Higher values penalize older matches heavily.") if mode == "Historical CSV" else None
    dc_rho = st.slider("Dixon-Coles Adjustment Factor", -0.15, 0.15, -0.05, step=0.01, help="Adjusts low-scoring drawing trends based on history.")
    max_goals = st.slider("Maximum goals per team in score grid", 6, 15, 10)
    uploaded = st.file_uploader("Upload historical results CSV", type=["csv"]) if mode == "Historical CSV" else None

df = None
hxg = axg = None
meta = {}

if mode == "Historical CSV":
    st.subheader("Historical data mode (Pro)")
    st.write("Upload a CSV file containing past match results to calibrate the model.")
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            
            # Smart normalization map for common external soccer datasets
            rename_map = {}
            for col in df.columns:
                c_clean = str(col).strip().lower()
                if c_clean in ["home_team", "hometeam", "home", "ht"]:
                    rename_map[col] = "home_team"
                elif c_clean in ["away_team", "awayteam", "away", "at"]:
                    rename_map[col] = "away_team"
                elif c_clean in ["home_goals", "fthg", "homegoals", "hg"]:
                    rename_map[col] = "home_goals"
                elif c_clean in ["away_goals", "ftag", "awaygoals", "ag"]:
                    rename_map[col] = "away_goals"
                elif c_clean in ["date", "match_date", "dt"]:
                    rename_map[col] = "date"
            
            df = df.rename(columns=rename_map)
            REQUIRED = {"home_team", "away_team", "home_goals", "away_goals"}
            missing = REQUIRED - set(df.columns)
            
            if missing:
                st.error("Missing required fields. Please ensure your CSV contains home/away team columns and score columns.")
            else:
                hxg, axg, meta = estimate_xg_weighted(df, home_team, away_team, lookback, time_decay)
                st.success(f"Successfully processed {meta['matches_in_file']} rows with time-decay form weights.")
        except Exception as e:
            st.error(f"Could not process that CSV: {e}")
    else:
        st.info("Upload a CSV of completed matches to run the advanced form calculator.")
else:
    st.subheader("Manual expected-goals mode")
    c1, c2 = st.columns(2)
    hxg = c1.number_input(f"{home_team} expected goals", min_value=0.15, max_value=4.50, value=1.55, step=0.05)
    axg = c2.number_input(f"{away_team} expected goals", min_value=0.15, max_value=4.50, value=1.10, step=0.05)

if hxg is not None and axg is not None and home_team.strip() and away_team.strip() and home_team.strip().casefold() != away_team.strip().casefold():
    result = calculate(float(hxg), float(axg), max_goals, dc_rho)
    st.divider()
    st.subheader(f"{home_team} vs {away_team}")
    st.caption(f"Model run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} • Dixon-Coles Pro Poisson model")
    st.caption(f"Expected goals used: {home_team} {hxg:.2f} · {away_team} {axg:.2f}")
    
    a, b, c = st.columns(3)
    a.metric(f"{home_team} win", f"{result['home_win']:.1%}")
    b.metric("Draw", f"{result['draw']:.1%}")
    c.metric(f"{away_team} win", f"{result['away_win']:.1%}")
    
    st.subheader("Goal markets")
    markets = [
        ("> 1.5 total goals", result["over15"]), ("> 2.5 total goals", result["over25"]),
        ("> 3.5 total goals", result["over35"]), ("Both teams to score", result["btts"]),
        (f"{home_team} to score 1+", result["home_score"]), (f"{away_team} to score 1+", result["away_score"])
    ]
    cols = st.columns(3)
    for i, (label, p) in enumerate(markets):
        cols[i%3].metric(label, f"{p:.1%}")


    
