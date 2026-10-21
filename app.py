import math
from datetime import datetime, timezone
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MatchLab 2.0", page_icon="⚽", layout="wide")
st.title("⚽ MatchLab 2.0")
st.caption("Football probability lab • historical-data mode + manual-input mode")

st.warning(
    "Prototype notice: this app does not have a live sports feed built in. "
    "Its output is only as good as the historical data or expected-goals inputs you provide. "
    "Probabilities are estimates, not guarantees."
)

REQUIRED = {"home_team", "away_team", "home_goals", "away_goals"}

def poisson_pmf(k, lam):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))

def calculate(home_xg, away_xg, max_goals=10):
    max_goals = max(6, min(int(max_goals), 15))
    hp = [poisson_pmf(i, home_xg) for i in range(max_goals + 1)]
    ap = [poisson_pmf(i, away_xg) for i in range(max_goals + 1)]
    scores = {(h, a): hp[h] * ap[a] for h in range(len(hp)) for a in range(len(ap))}
    mass = sum(scores.values())
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

def estimate_xg(df, home_name, away_name, lookback=10):
    """Estimate each team's expected goals using recent home/away results and league averages."""
    d = df.copy()
    d["home_team"] = d["home_team"].astype(str).str.strip()
    d["away_team"] = d["away_team"].astype(str).str.strip()
    for col in ["home_goals", "away_goals"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["home_team", "away_team", "home_goals", "away_goals"])
    d = d[(d.home_goals >= 0) & (d.away_goals >= 0)]
    if len(d) < 10:
        raise ValueError("Please upload at least 10 completed matches for a basic estimate.")
    if "date" in d.columns:
        d["_date"] = pd.to_datetime(d["date"], errors="coerce", utc=True)
        d = d.sort_values("_date")
    home_matches = d[d["home_team"].str.casefold() == home_name.strip().casefold()].tail(lookback)
    away_matches = d[d["away_team"].str.casefold() == away_name.strip().casefold()].tail(lookback)
    home_all = d[(d["home_team"].str.casefold() == home_name.strip().casefold()) |
                 (d["away_team"].str.casefold() == home_name.strip().casefold())].tail(lookback)
    away_all = d[(d["home_team"].str.casefold() == away_name.strip().casefold()) |
                 (d["away_team"].str.casefold() == away_name.strip().casefold())].tail(lookback)
    if home_matches.empty:
        raise ValueError(f"No home matches found for '{home_name}' in the uploaded file.")
    if away_matches.empty:
        raise ValueError(f"No away matches found for '{away_name}' in the uploaded file.")
    league_home = d["home_goals"].mean()
    league_away = d["away_goals"].mean()
    # Shrink small samples toward league averages to reduce overreaction.
    def mean_or(series, fallback):
        return float(series.mean()) if len(series) else fallback
    h_scored = mean_or(home_matches["home_goals"], league_home)
    h_conceded = mean_or(home_matches["away_goals"], league_away)
    a_scored = mean_or(away_matches["away_goals"], league_away)
    a_conceded = mean_or(away_matches["home_goals"], league_home)
    # Blend venue-specific performance with overall team performance.
    h_overall_scored = mean_or(
        pd.concat([home_all.loc[home_all.home_team.str.casefold() == home_name.casefold(), "home_goals"],
                   home_all.loc[home_all.away_team.str.casefold() == home_name.casefold(), "away_goals"]]),
        league_home
    )
    a_overall_scored = mean_or(
        pd.concat([away_all.loc[away_all.home_team.str.casefold() == away_name.casefold(), "home_goals"],
                   away_all.loc[away_all.away_team.str.casefold() == away_name.casefold(), "away_goals"]]),
        league_away
    )
    h_attack = 0.65 * h_scored + 0.35 * h_overall_scored
    a_attack = 0.65 * a_scored + 0.35 * a_overall_scored
    home_xg = (h_attack / max(league_home, 0.1)) * (a_conceded / max(league_home, 0.1)) * league_home
    away_xg = (a_attack / max(league_away, 0.1)) * (h_conceded / max(league_away, 0.1)) * league_away
    return max(0.15, min(home_xg, 4.5)), max(0.15, min(away_xg, 4.5)), {
        "matches_in_file": len(d), "home_venue_sample": len(home_matches),
        "away_venue_sample": len(away_matches), "league_home_avg": league_home,
        "league_away_avg": league_away
    }

with st.sidebar:
    st.header("1. Select match")
    home_team = st.text_input("Home team", "Arsenal")
    away_team = st.text_input("Away team", "Newcastle")
    mode = st.radio("How should expected goals be estimated?", ["Historical CSV", "Enter expected goals manually"])
    st.header("2. Model settings")
    lookback = st.slider("Recent venue matches to use", 5, 20, 10) if mode == "Historical CSV" else 10
    max_goals = st.slider("Maximum goals per team in score grid", 6, 15, 10)
    uploaded = st.file_uploader("Upload historical results CSV", type=["csv"]) if mode == "Historical CSV" else None

df = None
hxg = axg = None
meta = {}
if mode == "Historical CSV":
    st.subheader("Historical data")
    st.write("CSV must contain these columns: `home_team`, `away_team`, `home_goals`, `away_goals`. A `date` column is recommended.")
    st.write("Example row: `2026-09-20,Arsenal,Newcastle,2,1`")
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            missing = REQUIRED - set(df.columns)
            if missing:
                st.error("Missing required columns: " + ", ".join(sorted(missing)))
            else:
                hxg, axg, meta = estimate_xg(df, home_team, away_team, lookback)
                st.success(f"Loaded {meta['matches_in_file']} match rows. Estimated expected goals from your file.")
        except Exception as e:
            st.error(f"Could not process that CSV: {e}")
    else:
        st.info("Upload a CSV of completed matches. Without historical data, use manual mode to try the calculator.")
else:
    st.subheader("Manual expected-goals mode")
    st.caption("Use only if you already have defensible expected-goals estimates. These numbers are not fetched from live data.")
    c1, c2 = st.columns(2)
    hxg = c1.number_input(f"{home_team} expected goals", min_value=0.15, max_value=4.50, value=1.55, step=0.05)
    axg = c2.number_input(f"{away_team} expected goals", min_value=0.15, max_value=4.50, value=1.10, step=0.05)

if hxg is not None and axg is not None and home_team.strip() and away_team.strip() and home_team.strip().casefold() != away_team.strip().casefold():
    result = calculate(float(hxg), float(axg), max_goals)
    st.divider()
    st.subheader(f"{home_team} vs {away_team}")
    st.caption(f"Model run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} • Independent Poisson score model")
    st.caption(f"Expected goals used: {home_team} {hxg:.2f} · {away_team} {axg:.2f}")
    a,b,c = st.columns(3)
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
    for i,(label,p) in enumerate(markets):
        cols[i%3].metric(label, f"{p:.1%}")
    st.subheader("Most likely scorelines")
    top_df = pd.DataFrame(result["top"], columns=["Scoreline", "Probability"])
    top_df["Probability"] = top_df["Probability"].map(lambda x: f"{x:.2%}")
    st.dataframe(top_df, hide_index=True, use_container_width=True)
    if meta:
        st.subheader("Data coverage used")
        st.write(f"Home-team home sample: {meta['home_venue_sample']} matches; away-team away sample: {meta['away_venue_sample']} matches.")
        st.write(f"League averages in file: home goals {meta['league_home_avg']:.2f}; away goals {meta['league_away_avg']:.2f}.")
    st.info("Reliability warning: these are model-based estimates, not certainty. The CSV mode is a simple baseline, not a fully trained or backtested AI model. It does not account for lineups, injuries, red cards, odds, schedule congestion, or xG unless reflected in the input data.")
    st.download_button(
        "Download this prediction as CSV",
        pd.DataFrame([{
            "run_utc": datetime.now(timezone.utc).isoformat(),
            "home_team": home_team, "away_team": away_team,
            "home_expected_goals": hxg, "away_expected_goals": axg,
            "home_win_probability": result["home_win"], "draw_probability": result["draw"],
            "away_win_probability": result["away_win"], "over_1_5": result["over15"],
            "over_2_5": result["over25"], "over_3_5": result["over35"],
            "btts": result["btts"], "home_to_score": result["home_score"],
            "away_to_score": result["away_score"]
        }]).to_csv(index=False),
        file_name="matchlab_prediction.csv", mime="text/csv"
    )
elif home_team.strip().casefold() == away_team.strip().casefold():
    st.error("Choose two different teams.")

st.divider()
st.subheader("How to judge reliability")
st.markdown(
    "- Save predictions **before kickoff** and compare them with final results.\n"
    "- One match cannot establish reliability; evaluate at least hundreds of out-of-sample matches.\n"
    "- Track Brier score/log loss and calibration, not just how many top picks win.\n"
    "- Never describe a high probability as a guaranteed result."
)
