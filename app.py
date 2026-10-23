
import json
import math
import os
import random
import re
import sqlite3
import time
import html
from datetime import date, datetime, timedelta
from pathlib import Path
from collections import Counter
from html.parser import HTMLParser
from urllib.parse import quote_plus

import pandas as pd
import requests
import streamlit as st

# ============================================================
# MATCHLAB LIVE — ADAPTIVE v3
# Free public football data + live Internet research
# ============================================================

st.set_page_config(
    page_title="MATCHLAB LIVE",
    page_icon="⚽",
    layout="wide",
)

DB_PATH = Path("matchlab_history.db")
TSDB_BASE = "https://www.thesportsdb.com/api/v1/json/123"

st.markdown(
    """
    <style>
    :root {
        --ml-navy:#071A33;
        --ml-navy2:#0B2547;
        --ml-yellow:#F4C542;
        --ml-yellow2:#FFD95A;
        --ml-text:#F4F7FB;
        --ml-muted:#AAB8CC;
    }
    .stApp {background:linear-gradient(180deg,#061426 0%,#081B34 55%,#071A33 100%);color:var(--ml-text);}
    [data-testid="stSidebar"] {background:linear-gradient(180deg,#061426,#0A2342);border-right:1px solid rgba(244,197,66,.22);}
    [data-testid="stSidebar"] * {color:#F4F7FB;}
    .title {font-size:42px;font-weight:850;margin-bottom:0;color:#F4F7FB;}
    .subtitle,.muted {color:var(--ml-muted);}
    .welcome {padding:18px;border-radius:16px;background:rgba(11,37,71,.72);border:1px solid rgba(244,197,66,.22);margin-bottom:20px;}
    .reason {padding:14px 16px;border-radius:13px;margin-bottom:10px;background:rgba(11,37,71,.62);border:1px solid rgba(244,197,66,.14);}
    .hero {padding:28px;border-radius:24px;background:linear-gradient(135deg,#0B2547,#102F58);border:1px solid rgba(244,197,66,.28);margin:8px 0 22px;box-shadow:0 12px 35px rgba(0,0,0,.20);}
    .hero h1 {font-size:46px;letter-spacing:-1.2px;margin:0;color:#F4F7FB;}
    .hero p {margin:7px 0 0;color:#B9C7D9;font-size:15px;}
    .signal {padding:14px 16px;border-radius:16px;background:rgba(11,37,71,.62);border:1px solid rgba(244,197,66,.16);}
    .decision {padding:20px;border-radius:18px;background:linear-gradient(135deg,rgba(244,197,66,.15),rgba(11,37,71,.86));border:1px solid rgba(244,197,66,.42);font-size:18px;}
    .decision strong,.yellow {color:var(--ml-yellow);}
    .ask-box {padding:18px;border-radius:18px;background:#0B2547;border:1px solid rgba(244,197,66,.28);}
    .tag {display:inline-block;padding:5px 10px;border-radius:999px;background:rgba(244,197,66,.13);border:1px solid rgba(244,197,66,.28);color:#FFD95A;font-size:12px;font-weight:700;margin-right:6px;}
    div.stButton > button[kind="primary"] {background:#F4C542;color:#071A33;border:none;font-weight:800;}
    div.stButton > button[kind="primary"]:hover {background:#FFD95A;color:#071A33;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# GENERAL HELPERS
# ============================================================

def clamp(value, low, high):
    return max(low, min(high, value))

def norm(value):
    return " ".join(str(value or "").strip().lower().split())

def pct(value):
    return f"{value * 100:.1f}%"

def safe_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default

def parse_date(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None

# ============================================================
# SQLITE HISTORY
# ============================================================

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                fixture_id TEXT,
                match_date TEXT,
                competition TEXT,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                home_probability REAL,
                draw_probability REAL,
                away_probability REAL,
                home_xg REAL,
                away_xg REAL,
                predicted_outcome TEXT,
                predicted_score TEXT,
                confidence_score REAL,
                model_source TEXT,
                actual_home_goals INTEGER,
                actual_away_goals INTEGER,
                result_checked INTEGER DEFAULT 0,
                value_summary TEXT,
                reasons_json TEXT,
                raw_json TEXT
            )
        """)
        conn.commit()

init_db()

def prediction_exists(fixture_id, match_date):
    with sqlite3.connect(DB_PATH) as conn:
        row=conn.execute("SELECT id FROM predictions WHERE fixture_id=? AND match_date=? ORDER BY id DESC LIMIT 1",(str(fixture_id),str(match_date))).fetchone()
    return row[0] if row else None

def save_prediction(record):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO predictions (
                created_at, fixture_id, match_date, competition,
                home_team, away_team,
                home_probability, draw_probability, away_probability,
                home_xg, away_xg, predicted_outcome, predicted_score,
                confidence_score, model_source, value_summary,
                reasons_json, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            record.get("fixture_id"),
            record.get("match_date"),
            record.get("competition"),
            record["home_team"], record["away_team"],
            record["home_probability"], record["draw_probability"],
            record["away_probability"], record["home_xg"], record["away_xg"],
            record["predicted_outcome"], record["predicted_score"],
            record.get("confidence_score"), record.get("model_source"),
            record.get("value_summary", ""),
            json.dumps(record.get("reasons", [])),
            json.dumps(record, default=str),
        ))
        conn.commit()

def load_history(limit=100):
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("""
            SELECT id, created_at, fixture_id, match_date, competition,
                   home_team, away_team, home_probability, draw_probability,
                   away_probability, home_xg, away_xg, predicted_outcome,
                   predicted_score, confidence_score, model_source,
                   actual_home_goals, actual_away_goals, result_checked,
                   value_summary
            FROM predictions ORDER BY id DESC LIMIT ?
        """, conn, params=(limit,))

def delete_prediction(prediction_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM predictions WHERE id = ?", (int(prediction_id),))
        conn.commit()

def update_actual_result(prediction_id, home_goals, away_goals):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            UPDATE predictions
            SET actual_home_goals=?, actual_away_goals=?, result_checked=1
            WHERE id=?
        """, (home_goals, away_goals, int(prediction_id)))
        conn.commit()


def load_learning_rows(target_date=None):
    """Only use completed predictions made before the target fixture. Prevents leakage."""
    with sqlite3.connect(DB_PATH) as conn:
        rows=conn.execute("""SELECT match_date,home_xg,away_xg,actual_home_goals,actual_away_goals,
                                   home_probability,draw_probability,away_probability,raw_json
                            FROM predictions WHERE result_checked=1 AND actual_home_goals IS NOT NULL
                            ORDER BY id DESC LIMIT 1000""").fetchall()
    out=[]
    for row in rows:
        try:
            d=str(row[0])[:10]
            if target_date and d >= str(target_date):
                continue
            out.append(row)
        except Exception:
            continue
    return out


def calibrate_1x2(matrix_result, target_date=None):
    """Calibrate 1X2 probabilities using completed MATCHLAB predictions.

    The calibration is deliberately conservative: it only activates after enough
    completed matches and blends the model with empirical outcome frequencies.
    This protects the draw probability from being crushed by an overconfident
    favourite, one of the lessons from recent MATCHLAB tests.
    """
    rows=load_learning_rows(target_date)
    if len(rows)<12:
        return dict(matrix_result), {
            "samples":len(rows), "active":False, "weight":0.0,
            "status":"Calibration warm-up: need at least 12 completed predictions."
        }
    actual_counts={"home":0,"draw":0,"away":0}
    pred_sum={"home":0.0,"draw":0.0,"away":0.0}
    usable=0
    for r in rows:
        try:
            hg,ag=int(r[3]),int(r[4])
            outcome="home" if hg>ag else "draw" if hg==ag else "away"
            actual_counts[outcome]+=1
            pred_sum["home"]+=float(r[5]); pred_sum["draw"]+=float(r[6]); pred_sum["away"]+=float(r[7])
            usable+=1
        except Exception:
            continue
    if usable<12:
        return dict(matrix_result), {"samples":usable,"active":False,"weight":0.0,"status":"Not enough usable completed predictions for calibration."}
    # Laplace smoothing avoids overreacting to small samples.
    empirical={k:(actual_counts[k]+1)/(usable+3) for k in actual_counts}
    mean_pred={k:pred_sum[k]/usable for k in pred_sum}
    # At most 30% empirical correction. Increase slowly with sample size.
    weight=min(0.30, 0.08 + usable/500.0)
    raw={
        "home_win":float(matrix_result.get("home_win",1/3)),
        "draw":float(matrix_result.get("draw",1/3)),
        "away_win":float(matrix_result.get("away_win",1/3)),
    }
    adjusted={
        "home_win":(1-weight)*raw["home_win"]+weight*empirical["home"],
        "draw":(1-weight)*raw["draw"]+weight*empirical["draw"],
        "away_win":(1-weight)*raw["away_win"]+weight*empirical["away"],
    }
    total=sum(adjusted.values()) or 1.0
    for k in adjusted: adjusted[k]/=total
    out=dict(matrix_result); out.update(adjusted)
    return out,{
        "samples":usable,"active":True,"weight":weight,
        "empirical":empirical,"mean_predicted":mean_pred,
        "status":"Online 1X2 calibration active; empirical results blended conservatively."
    }


def learning_metrics(target_date=None):
    """Return calibration diagnostics for completed predictions."""
    rows=load_learning_rows(target_date)
    if not rows:
        return {"samples":0}
    brier=[]; acc=[]; xg_abs=[]; total_goal_abs=[]
    for r in rows:
        try:
            hp,dp,ap=float(r[5]),float(r[6]),float(r[7])
            hg,ag=int(r[3]),int(r[4])
            actual=[1.0,0.0,0.0] if hg>ag else [0.0,1.0,0.0] if hg==ag else [0.0,0.0,1.0]
            brier.append((hp-actual[0])**2+(dp-actual[1])**2+(ap-actual[2])**2)
            pred_idx=max(range(3),key=lambda i:[hp,dp,ap][i])
            actual_idx=0 if hg>ag else 1 if hg==ag else 2
            acc.append(1 if pred_idx==actual_idx else 0)
            xg_abs.append(abs(float(r[1])-hg)+abs(float(r[2])-ag))
            total_goal_abs.append(abs((float(r[1])+float(r[2]))-(hg+ag)))
        except Exception:
            continue
    if not brier: return {"samples":0}
    return {
        "samples":len(brier),
        "brier":sum(brier)/len(brier),
        "accuracy":sum(acc)/len(acc),
        "xg_mae_total":sum(xg_abs)/len(xg_abs),
        "total_goals_mae":sum(total_goal_abs)/len(total_goal_abs),
    }


def safest_double_chance(matrix_result):
    """Return the safest 1X2-derived double-chance outcome without using odds."""
    options={
        "1X":matrix_result["home_win"]+matrix_result["draw"],
        "X2":matrix_result["draw"]+matrix_result["away_win"],
        "12":matrix_result["home_win"]+matrix_result["away_win"],
    }
    market=max(options,key=options.get)
    return market,options[market],options

def choose_primary_decision(matrix_result, home_team, away_team, coverage, uncertainty, provisional=False, style="Balanced"):
    """Choose one actionable football call without bookmaker odds or exact scores."""
    home=matrix_result.get("home_win",0.0)
    draw=matrix_result.get("draw",0.0)
    away=matrix_result.get("away_win",0.0)
    candidates=[(home, f"{home_team} to win", "RESULT"), (draw, "Match to end in a draw", "RESULT"), (away, f"{away_team} to win", "RESULT")]
    top_prob,top_label,top_type=max(candidates,key=lambda x:x[0])
    vals=[home,draw,away]
    gap=top_prob-sorted(vals)[-2]
    min_prob=0.49 if style=="Balanced" else 0.53
    min_gap=0.045 if style=="Balanced" else 0.075
    if not provisional and coverage>=55 and uncertainty<=0.26 and top_prob>=min_prob and gap>=min_gap:
        return {"decision":top_label,"type":"RESULT","probability":top_prob,"reason":"The strongest 1X2 outcome clears MATCHLAB's evidence, probability and separation thresholds."}

    goals=[
        (matrix_result.get("under_35",0.0),"Under 3.5 goals","GOALS"),
        (matrix_result.get("over_15",0.0),"Over 1.5 goals","GOALS"),
        (matrix_result.get("under_25",0.0),"Under 2.5 goals","GOALS"),
        (matrix_result.get("over_25",0.0),"Over 2.5 goals","GOALS"),
        (matrix_result.get("btts_yes",0.0),"Both teams to score","GOALS"),
        (matrix_result.get("btts_no",0.0),"Both teams NOT to score","GOALS"),
    ]
    goals.sort(reverse=True,key=lambda x:x[0])
    gp,gl,gt=goals[0]
    # Prefer a goal market when the result is structurally close.
    if gp>=0.62:
        return {"decision":gl,"type":"GOALS","probability":gp,"reason":"The 1X2 contest is not separated enough, so MATCHLAB selects the strongest supported goals market."}
    # Never return NO BET just because the game is close: choose the highest-probability 1X2 outcome and label it as a lean.
    return {"decision":top_label,"type":"RESULT","probability":top_prob,"reason":"No goals market cleared the stronger threshold, so MATCHLAB keeps the highest-probability 1X2 outcome as the final lean."}

def matrix_simulation_stability(matrix_result, simulation_result):
    """Compare deterministic probability matrix with Monte Carlo as a stability check."""
    if not simulation_result:
        return {"available":False,"max_gap":None,"status":"Simulation unavailable."}
    pairs=[
        abs(matrix_result.get("home_win",0)-simulation_result.get("home_win",0)),
        abs(matrix_result.get("draw",0)-simulation_result.get("draw",0)),
        abs(matrix_result.get("away_win",0)-simulation_result.get("away_win",0)),
        abs(matrix_result.get("over_25",0)-simulation_result.get("over_25",0)),
        abs(matrix_result.get("btts_yes",0)-simulation_result.get("btts_yes",0)),
    ]
    gap=max(pairs) if pairs else 0.0
    return {"available":True,"max_gap":gap,"status":"Stable" if gap<0.025 else "Minor simulation variance" if gap<0.05 else "High model dispersion"}


# ============================================================
# INTERACTIVE RESEARCH AGENT
# ============================================================

TRUSTED_SOURCE_HINTS = {
    "official": ["official", "club", "fc.com", "afcb", "mancity", "arsenal.com", "chelseafc", "liverpoolfc", "realmadrid", "fcbarcelona"],
    "high": ["reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "espn.com", "theguardian.com", "skysports.com", "uefa.com", "fifa.com"],
    "football": ["transfermarkt", "soccerway", "fotmob", "flashscore", "worldfootball", "besoccer", "sofascore"],
}

def _source_quality(url, title=""):
    text=norm(f"{url} {title}")
    if any(x in text for x in TRUSTED_SOURCE_HINTS["official"]): return 4
    if any(x in text for x in TRUSTED_SOURCE_HINTS["high"]): return 3
    if any(x in text for x in TRUSTED_SOURCE_HINTS["football"]): return 2
    return 1

def _research_intent(question):
    q=norm(question)
    intents=[]
    groups={
        "team_news":["injur","suspension","available","missing","lineup","xi","squad","fitness","ban","return"],
        "form":["form","recent","last 5","last five","results","home","away","attack","defence","defense"],
        "prediction":["why","winner","win","draw","result","prediction","pick","call","best","strong","confidence"],
        "goals":["goal","over","under","btts","both teams","scoring","clean sheet"],
        "motivation":["motivation","table","promotion","relegation","title","europe","playoff","must win","importance"],
        "tactics":["tactic","formation","style","press","counter","possession","matchup","manager"],
        "weather":["weather","rain","wind","pitch","temperature","condition"],
        "schedule":["rest","fatigue","fixture","congestion","travel","rotation","next match","midweek"],
        "h2h":["head to head","h2h","previous meeting","historical"],
        "coverage":["source","data","coverage","reliable","latest","current","today","now"],
    }
    for name,terms in groups.items():
        if any(t in q for t in terms): intents.append(name)
    if not intents: intents=["prediction","team_news","form","goals","coverage"]
    return intents

def _build_research_plan(home, away, match_date, competition, question):
    """Build a broad but purposeful live-research plan from the user's question."""
    base=f'"{home}" "{away}" {competition} {match_date}'
    intents=_research_intent(question)
    queries=[]
    # Always establish current fixture context and latest reporting.
    queries += [
        f'{base} latest preview team news',
        f'"{home}" latest news {match_date} {competition}',
        f'"{away}" latest news {match_date} {competition}',
    ]
    templates={
        "team_news":[
            f'{base} confirmed lineup injuries suspensions squad news',
            f'"{home}" confirmed lineup injuries suspension latest',
            f'"{away}" confirmed lineup injuries suspension latest',
            f'{base} official team news lineup',
        ],
        "form":[f'{base} last five form results home away goals', f'"{home}" home form recent results', f'"{away}" away form recent results'],
        "prediction":[f'{base} prediction preview odds analysis', f'{base} expert preview likely result', f'{base} statistical prediction model'],
        "goals":[f'{base} over under goals BTTS statistics', f'{base} expected goals scoring defence'],
        "motivation":[f'{base} league table promotion relegation motivation', f'{base} must win importance manager comments'],
        "tactics":[f'{base} tactical preview formation pressing matchup manager', f'{base} tactical analysis style of play'],
        "weather":[f'{base} weather stadium forecast wind rain', f'"{home}" stadium weather {match_date}'],
        "schedule":[f'{base} rest days fixture congestion rotation travel', f'"{home}" next fixture rotation {match_date}', f'"{away}" next fixture rotation {match_date}'],
        "h2h":[f'{base} head to head previous meetings', f'"{home}" "{away}" historical head to head'],
        "coverage":[f'{base} latest official announcement', f'{base} today confirmed team news'],
    }
    for intent in intents:
        queries.extend(templates.get(intent,[]))
    # Ask the exact user question explicitly, plus a neutral variant to reduce search bias.
    queries += [f'{base} {question}', f'"{home}" "{away}" {question}', f'{base} latest update']
    out=[]; seen=set()
    for q in queries:
        k=norm(q)
        if k and k not in seen:
            seen.add(k); out.append(q)
    return out[:18], intents

@st.cache_data(ttl=300, show_spinner=False)
def research_agent(home_name, away_name, match_date, competition_hint, question):
    """Multi-query live research agent.

    It does not blindly search once. It plans queries from intent, searches both
    public engines, scores source quality, removes duplicates, detects obvious
    conflicting snippets, and returns an evidence packet for the conversational
    analyst. No paid API key is required.
    """
    plan,intents=_build_research_plan(home_name,away_name,match_date,competition_hint,question)
    results=[]; seen=set(); query_log=[]
    for search_q in plan:
        try:
            found=public_web_search(search_q,max_results=6)
        except Exception:
            found=[]
        query_log.append({"query":search_q,"results":len(found)})
        for r in found:
            url=str(r.get("url") or "").strip()
            title=str(r.get("title") or "").strip()
            key=(url or title).lower()
            if not key or key in seen: continue
            seen.add(key)
            item=dict(r)
            item["quality"]=_source_quality(url,title)
            item["query"] = search_q
            results.append(item)
    # Quality first, then preserve some diversity rather than returning one site's results only.
    results.sort(key=lambda r:(r.get("quality",1), len(str(r.get("content", "")))), reverse=True)
    results=results[:48]

    # Very lightweight conflict detection for clear availability/score words.
    conflict_terms=("out","injured","suspended","available","fit","ruled out","starts","benched","wins","draw")
    topic_hits={}
    for r in results:
        txt=norm(f"{r.get('title','')} {r.get('content','')}")
        for term in conflict_terms:
            if term in txt:
                topic_hits.setdefault(term,[]).append(r)
    conflicts=[]
    for term,items in topic_hits.items():
        # A conflict is flagged when multiple distinct sources discuss the same term.
        urls={i.get('url') for i in items if i.get('url')}
        if len(urls)>=3:
            conflicts.append({"topic":term,"sources":len(urls)})

    return {
        "agent":"MATCHLAB Research Agent",
        "question":question,
        "intents":intents,
        "plan":plan,
        "query_log":query_log,
        "results":results,
        "conflicts":conflicts[:10],
        "source_count":len(results),
        "searched_at":datetime.utcnow().isoformat(timespec="seconds")+"Z",
        "method":"DuckDuckGo HTML + Google News RSS; source-quality ranking; deduplication; conflict flags",
    }


def _global_football_research_plan(question):
    """Plan web research for ANY football question, not only the active fixture."""
    q=norm(question)
    today=date.today().isoformat()
    intents=[]
    if any(k in q for k in ["play tomorrow","playing tomorrow","next match","next game","fixture","fixtures","schedule","when do","kickoff","kick off"]):
        intents.append("fixtures")
    if any(k in q for k in ["score","result","live","tonight","today","yesterday","last night"]):
        intents.append("scores")
    if any(k in q for k in ["injur","lineup","starting xi","squad","suspension","available","missing","team news"]):
        intents.append("team_news")
    if any(k in q for k in ["transfer","signed","signing","rumour","rumor","contract"]):
        intents.append("transfers")
    if any(k in q for k in ["table","standings","position","points","relegation","promotion","title"]):
        intents.append("table")
    if any(k in q for k in ["player","striker","goals","assists","stats","statistic","scored","form"]):
        intents.append("player_stats")
    if any(k in q for k in ["manager","coach","tactics","formation","pressing","style"]):
        intents.append("tactics")
    if any(k in q for k in ["head to head","h2h","history","record"]):
        intents.append("h2h")
    if any(k in q for k in ["league","competition","champions league","europa","conference league","premier league","laliga","bundesliga","serie a","ligue 1"]):
        intents.append("competition")
    if any(k in q for k in ["why","predict","prediction","will","likely","winner","draw","over","under","btts","bet"]):
        intents.append("analysis")
    if not intents:
        intents=["general_football"]

    templates={
        "fixtures":[f'football fixtures {question} {today}', f'{question} football fixture latest', f'{question} official schedule'],
        "scores":[f'football scores {question} latest {today}', f'{question} football live score result', f'{question} latest match result'],
        "team_news":[f'{question} latest team news injury lineup', f'{question} official team news', f'{question} confirmed lineup squad'],
        "transfers":[f'{question} transfer latest football', f'{question} official transfer announcement', f'{question} transfer news latest'],
        "table":[f'{question} football table standings {today}', f'{question} league table latest', f'{question} points position football'],
        "player_stats":[f'{question} football player stats latest', f'{question} goals assists appearances football', f'{question} player form statistics'],
        "tactics":[f'{question} tactical analysis football', f'{question} manager formation tactics latest'],
        "h2h":[f'{question} head to head football', f'{question} previous meetings results'],
        "competition":[f'{question} football competition latest', f'{question} official competition information'],
        "analysis":[f'{question} football analysis prediction latest', f'{question} football preview statistics', f'{question} latest expert analysis'],
        "general_football":[f'{question} football', f'{question} soccer latest', f'{question} football news {today}'],
    }
    queries=[]; seen=set()
    for intent in intents:
        for qx in templates.get(intent,[]):
            k=norm(qx)
            if k not in seen:
                seen.add(k); queries.append(qx)
    queries += [question, f'{question} football latest']
    out=[]; seen=set()
    for qx in queries:
        k=norm(qx)
        if k and k not in seen:
            seen.add(k); out.append(qx)
    return out[:16], intents

@st.cache_data(ttl=180, show_spinner=False)
def global_football_research(question):
    """Fresh, fixture-independent public-web research for any football question."""
    plan,intents=_global_football_research_plan(question)
    results=[]; seen=set(); query_log=[]
    for search_q in plan:
        try:
            found=public_web_search(search_q,max_results=6)
        except Exception:
            found=[]
        query_log.append({"query":search_q,"results":len(found)})
        for r in found:
            url=str(r.get("url") or "").strip()
            title=str(r.get("title") or "").strip()
            key=(url or title).lower()
            if not key or key in seen: continue
            seen.add(key)
            item=dict(r)
            item["quality"]=_source_quality(url,title)
            item["query"]=search_q
            results.append(item)
    results.sort(key=lambda r:(r.get("quality",1),len(str(r.get("content", "")))),reverse=True)
    results=results[:50]
    # Detect obvious contradictory reporting without pretending that snippets are ground truth.
    conflict_terms=("ruled out","available","starts","benched","injured","suspended","signed","wins","draw","postponed")
    conflicts=[]
    for term in conflict_terms:
        hits=[r for r in results if term in norm(f"{r.get('title','')} {r.get('content','')}")]
        urls={r.get('url') for r in hits if r.get('url')}
        if len(urls)>=3:
            conflicts.append({"topic":term,"sources":len(urls)})
    return {
        "agent":"MATCHLAB Global Football Research Agent",
        "question":question,
        "intents":intents,
        "plan":plan,
        "query_log":query_log,
        "results":results,
        "conflicts":conflicts[:10],
        "source_count":len(results),
        "searched_at":datetime.utcnow().isoformat(timespec="seconds")+"Z",
        "method":"Multi-query DuckDuckGo HTML + Google News RSS; source-quality ranking; deduplication; conflict flags",
    }

def answer_global_football_question(question, packet):
    """Conversational answer layer for football questions without an active fixture."""
    results=(packet or {}).get("results",[]) or []
    if not results:
        return ("I couldn't find reliable fresh public information for that question. "
                "Try adding the team, player, league or competition name and I'll research it again.")
    q=norm(question)
    rows=[]
    for r in results[:8]:
        title=str(r.get("title","")).strip()
        content=str(r.get("content","")).strip()
        if title or content:
            rows.append(f"• **{title[:180]}** — {content[:420]}")
    answer=(f"I researched **{question}** across {len(results)} deduplicated public football sources.\n\n"
            + "\n".join(rows))
    if packet.get("conflicts"):
        answer += "\n\n⚠️ **Source conflict:** some public reports disagree on one or more points. I would not treat the disputed detail as confirmed until an official club, league or competition source confirms it."
    answer += "\n\n**If you want, keep asking follow-ups.** I can research the next question using fresh football information rather than assuming you're asking about the last match you analyzed."
    return answer

def research_agent_context(packet,max_chars=11000):
    rows=[]
    for r in (packet or {}).get("results",[])[:30]:
        rows.append(
            f"[{r.get('quality',1)}/4] {r.get('title','')}\n"
            f"{r.get('content','')}\nURL: {r.get('url','')}\nSEARCH: {r.get('query','')}"
        )
    return "\n\n".join(rows)[:max_chars]

def answer_model_question(question, analysis, live_research=None):
    """Contextual MATCHLAB analyst.

    This is intentionally fixture-grounded: it explains the live prediction from
    the evidence MATCHLAB already collected instead of inventing facts. It can
    handle follow-up questions such as "why?", "what about the draw?", "should
    I take over 1.5?", and "what could make this wrong?".
    """
    q=norm(question)
    home=analysis["home_team"]["name"]
    away=analysis["away_team"]["name"]
    matrix=analysis["matrix"]
    hx=analysis["home_xg"]
    ax=analysis["away_xg"]
    rec=analysis.get("decision",{})
    hp=float(matrix.get("home_win",0.0))
    dp=float(matrix.get("draw",0.0))
    ap=float(matrix.get("away_win",0.0))
    tn=analysis.get("team_news",{}) or {}
    hpv=analysis.get("home_profile",{}) or {}
    apv=analysis.get("away_profile",{}) or {}
    reasons=analysis.get("reasons",[]) or []
    risks=analysis.get("risks",[]) or []
    weather=analysis.get("weather_data",{}) or {}
    calibration=analysis.get("calibration",{}) or {}
    learning=analysis.get("learned_bias",{}) or {}
    web_results=(live_research or {}).get("results",[]) or []

    def pct1(x):
        return f"{float(x)*100:.1f}%"

    def research_note(limit=3):
        if not web_results:
            return ""
        note=f"\n\n**Fresh research:** MATCHLAB checked {len(web_results)} deduplicated public sources for this question."
        conflicts=(live_research or {}).get("conflicts",[]) or []
        if conflicts:
            note += " Some public reports conflict, so those claims remain uncertain."
        rows=[]
        for r in web_results[:limit]:
            title=str(r.get("title","")).strip()
            content=str(r.get("content","")).strip()
            if title or content:
                rows.append(f"• **{title[:160]}** — {content[:260]}")
        if rows:
            note += "\n" + "\n".join(rows)
        return note

    def team_news_line(side, name):
        block=tn.get(side,{}) or {}
        count=block.get("evidence_count",0)
        items=block.get("evidence",[]) or []
        snippets=[]
        for item in items[:3]:
            if isinstance(item,dict):
                title=item.get("title") or item.get("text") or ""
            else:
                title=str(item)
            if title:
                snippets.append(str(title)[:180])
        if snippets:
            return f"**{name}:** {count} relevant news item(s). " + " | ".join(snippets)
        return f"**{name}:** {count} relevant team-news item(s) were found."

    # If the user asks for current information, answer from a fresh web search first.
    latest_terms=["latest","today","now","current","just","news","lineup","injur","suspension","available","missing","confirmed"]
    if web_results and any(k in q for k in latest_terms):
        source_lines=[]
        for r in web_results[:7]:
            title=str(r.get("title","")).strip(); content=str(r.get("content","")).strip()
            if title or content:
                source_lines.append(f"• **{title}** — {content[:300]}")
        return (f"I researched the live public web for your question about **{home} vs {away}**. "
                f"I found **{len(web_results)} relevant result(s)**. Here is what the current reporting says:\n\n"
                + "\n".join(source_lines) +
                "\n\nI treat this as fresh evidence, not automatic truth. Conflicting reports stay uncertain, and official club/competition information gets priority when identifiable.")

    # Result / decision questions
    if any(k in q for k in ["full time", "final result", "who will win", "winner", "result"]):
        return (
            f"My current full-time call is **{rec.get('decision','not available')}**. "
            f"That is based on **{pct1(rec.get('probability',0))}** for the selected outcome. "
            f"The 1X2 distribution is **{home} {pct1(hp)} / Draw {pct1(dp)} / {away} {pct1(ap)}**. "
            f"I would not treat that as certainty; the closest alternative is "
            f"**{home if hp>=ap else away if ap>=hp else 'the draw'}** / draw depending on the probabilities."
        )

    # Direct comparisons: "team A or team B", "why X not Y"
    if any(k in q for k in ["or", "versus", "vs", "rather than", "instead of"]):
        mentioned_home=home.lower() in q
        mentioned_away=away.lower() in q
        if ("draw" in q or "tie" in q) and (mentioned_home or mentioned_away):
            side=home if mentioned_home else away
            sidep=hp if mentioned_home else ap
            return (
                f"Between **{side}** and the **draw**, MATCHLAB currently gives "
                f"**{pct1(sidep)}** to {side} and **{pct1(dp)}** to the draw. "
                f"The difference is **{abs(sidep-dp)*100:.1f} percentage points**. "
                f"So I would choose **{side}** only if its lead is supported by the "
                f"coverage and evidence shown above; otherwise the draw remains a live alternative."
            )

    # "Why?" and evidence questions
    if q in {"why", "why?", "explain", "explain it", "how did you decide"} or any(k in q for k in ["why did", "why this", "why not", "reason", "because", "explain"]):
        evidence=[str(x) for x in reasons[:6] if x]
        if not evidence:
            evidence=["The model found insufficient structured reasons to list separately."]
        target=rec.get("decision","the final call")
        return (
            f"I chose **{target}** because the evidence points that way, not because of bookmaker odds. "
            f"The strongest factors were:\n\n" +
            "\n".join([f"• {x}" for x in evidence]) +
            f"\n\nThe main uncertainty is **{pct1(rec.get('probability',0))}** confidence in the selected outcome, "
            f"with the other 1X2 probabilities at {pct1(hp)}, {pct1(dp)}, and {pct1(ap)}."
            + (f"\n\nI also checked **{len(web_results)} fresh web result(s)** for this question." if web_results else "")
            + research_note()
        )

    # Goals-market questions: make an actual recommendation, not just dump figures.
    if any(k in q for k in ["over 1.5", "under 1.5", "over 2.5", "under 2.5", "over 3.5", "under 3.5", "btts", "both teams"]):
        requested=None
        label=None
        if "over 1.5" in q: requested=matrix.get("over_15",0.0); label="Over 1.5 goals"
        elif "under 1.5" in q: requested=matrix.get("under_15",0.0); label="Under 1.5 goals"
        elif "over 2.5" in q: requested=matrix.get("over_25",0.0); label="Over 2.5 goals"
        elif "under 2.5" in q: requested=matrix.get("under_25",0.0); label="Under 2.5 goals"
        elif "over 3.5" in q: requested=matrix.get("over_35",0.0); label="Over 3.5 goals"
        elif "under 3.5" in q: requested=matrix.get("under_35",0.0); label="Under 3.5 goals"
        elif "btts" in q or "both teams" in q:
            requested=matrix.get("btts_yes",0.0); label="Both teams to score"
        if requested is not None:
            if float(requested)>=0.62:
                verdict=f"**Yes — {label} is a MATCHLAB-supported goals choice.**"
            elif float(requested)>=0.55:
                verdict=f"**Possible, but not strong enough for me to prefer it over the main call.**"
            else:
                verdict=f"**No — {label} is too close to a coin flip for MATCHLAB to prefer it.**"
            return (
                f"{verdict}\n\n"
                f"Estimated goals: **{hx:.2f} {home}** and **{ax:.2f} {away}**. "
                f"{label}: **{pct1(requested)}**. "
                f"The current overall decision is **{rec.get('decision','not available')}**. "
                f"So I would not switch to the goals market unless its probability is clearly stronger."
            )

    # Broad goals question
    if any(k in q for k in ["goals", "scoring", "how many goals"]):
        return (
            f"MATCHLAB estimates **{hx:.2f} goals for {home}** and **{ax:.2f} for {away}**. "
            f"Over 1.5: **{pct1(matrix.get('over_15',0))}**; "
            f"Over 2.5: **{pct1(matrix.get('over_25',0))}**; "
            f"Under 2.5: **{pct1(matrix.get('under_25',0))}**; "
            f"BTTS: **{pct1(matrix.get('btts_yes',0))}**. "
            f"The model's recommendation is based on whichever outcome/market has the strongest evidence, "
            f"rather than automatically choosing a goals market."
        )

    # Team-news / availability
    if any(k in q for k in ["injur", "news", "lineup", "suspension", "available", "missing", "absent", "squad"]):
        return (
            f"{team_news_line('home',home)}\n\n"
            f"{team_news_line('away',away)}\n\n"
            "MATCHLAB caps the impact of news so a few uncertain reports cannot completely overturn "
            "the historical/statistical signal. Confirmed lineup information gets more weight than speculation."
        )

    # Form / home-away
    if any(k in q for k in ["form", "recent", "last five", "home record", "away record", "attack", "defence", "defense"]):
        return (
            f"Recent profile: **{home}** — {hpv.get('points',0):.2f} points/match, "
            f"{hpv.get('gf',0):.2f} goals scored and {hpv.get('ga',0):.2f} conceded; "
            f"**{away}** — {apv.get('points',0):.2f} points/match, "
            f"{apv.get('gf',0):.2f} scored and {apv.get('ga',0):.2f} conceded. "
            f"MATCHLAB also separates home/away performance rather than treating all recent matches equally."
        )

    # Risks / alternative outcome
    if any(k in q for k in ["wrong", "risk", "lose", "fail", "alternative", "upset", "could happen"]):
        risk_text=[str(x) for x in risks[:6] if x]
        if not risk_text:
            risk_text=["The main risk is that football scoring variance can overturn a moderate probability edge."]
        return (
            f"The call can fail. The main risks I see are:\n\n" +
            "\n".join([f"• {x}" for x in risk_text]) +
            f"\n\nThe draw probability is **{pct1(dp)}**, while the other team has **{pct1(ap if rec.get('decision','').startswith(home)==False else hp)}** depending on the selected side. "
            "That is why MATCHLAB should never describe a prediction as guaranteed."
        )

    # Weather / environment
    if any(k in q for k in ["weather", "rain", "wind", "pitch", "condition"]):
        if weather.get("available"):
            return (
                f"Match conditions: **{weather.get('label','Available')}**. "
                f"Wind **{weather.get('wind_kmh',0):.0f} km/h**, precipitation **{weather.get('precipitation_mm',0):.1f} mm**. "
                f"MATCHLAB applied an environment factor of **{weather.get('environment_factor',1.0):.3f}**."
            )
        return "MATCHLAB could not obtain reliable stadium-weather data, so it kept the weather effect neutral rather than guessing."

    # Learning / calibration
    if any(k in q for k in ["learn", "learning", "calibrat", "history", "improve", "past predictions"]):
        return (
            f"MATCHLAB has **{learning.get('samples',0)}** completed predictions contributing to goal learning "
            f"and **{calibration.get('samples',0)}** completed predictions in 1X2 calibration. "
            f"{calibration.get('status','Calibration status is not yet strong enough to draw conclusions.')}"
        )

    # Coverage / data questions
    if any(k in q for k in ["data", "source", "coverage", "league", "premier", "laliga", "la liga", "bundesliga", "ligue 1", "serie a"]):
        return (
            "MATCHLAB is designed for **worldwide fixture discovery**, not a fixed team list. "
            "It uses free structured football data plus public web research and can work across major leagues "
            "such as the Premier League, LaLiga, Serie A, Bundesliga and Ligue 1, as well as many other competitions. "
            "Coverage is fixture-dependent: if a free source cannot verify a particular lower-tier or obscure match, "
            "MATCHLAB will say so rather than inventing data."
        )

    # Conversational fallback: the agent has already researched the user's question.
    answer=(
        f"I'm MATCHLAB's fixture analyst for **{home} vs {away}**. "
        f"My current call is **{rec.get('decision','not available')}**. "
        "I researched your question across multiple public sources and can explain the evidence, conflicts, "
        "team news, form, goals, risks, or how the prediction could change. "
        "Ask naturally — for example, **'what changed today?'**, **'why not the draw?'**, "
        "**'check the latest lineup'**, or **'what could make you wrong?'**."
    )
    return answer + research_note()

def learned_goal_bias(target_date=None):
    rows=load_learning_rows(target_date)
    if len(rows)<8:
        return {"samples":len(rows),"home_factor":1.0,"away_factor":1.0,"status":"Learning warm-up: need at least 8 completed predictions."}
    h_res=[]; a_res=[]
    for r in rows:
        try:
            h_res.append(float(r[3])-float(r[1])); a_res.append(float(r[4])-float(r[2]))
        except Exception: pass
    if not h_res:
        return {"samples":0,"home_factor":1.0,"away_factor":1.0,"status":"No usable completed predictions."}
    # Shrink the learned correction strongly; learning should improve the model, not chase noise.
    h_bias=sum(h_res)/len(h_res); a_bias=sum(a_res)/len(a_res)
    strength=min(1.0,len(h_res)/50.0)
    hf=clamp(1.0+(h_bias/2.0)*strength,0.94,1.06)
    af=clamp(1.0+(a_bias/2.0)*strength,0.94,1.06)
    return {"samples":len(h_res),"home_factor":hf,"away_factor":af,
            "home_residual":h_bias,"away_residual":a_bias,"status":"Online learning active from completed MATCHLAB predictions."}

def sync_completed_predictions(limit=40):
    """Automatically close the feedback loop when TheSportsDB has the final score."""
    if st.session_state.get("matchlab_history_synced"):
        return 0
    changed=0
    with sqlite3.connect(DB_PATH) as conn:
        rows=conn.execute("SELECT id,fixture_id,match_date,home_team,away_team FROM predictions WHERE result_checked=0 ORDER BY id DESC LIMIT ?",(limit,)).fetchall()
    for pid,fid,mdate,home,away in rows:
        if str(mdate)[:10] >= date.today().isoformat():
            continue
        event=None
        try:
            if fid and str(fid).isdigit(): event=football.event(fid)
        except Exception: pass
        if not event:
            try:
                events=football.search_event(f"{home} {away}")
                for e in events:
                    if str(e.get("dateEvent",""))[:10]==str(mdate)[:10] and event_matches_fixture(e,home,away):
                        event=e; break
            except Exception: pass
        if event and event.get("intHomeScore") is not None and event.get("intAwayScore") is not None:
            update_actual_result(pid,int(event["intHomeScore"]),int(event["intAwayScore"])); changed+=1
    st.session_state["matchlab_history_synced"]=True
    return changed

# ============================================================
# FREE STRUCTURED FOOTBALL DATA — THE SPORTsDB V1
# ============================================================

class FreeFootballData:
    """Uses TheSportsDB's documented free V1 API (key 123)."""

    def get(self, endpoint, params=None, timeout=15):
        url = TSDB_BASE + endpoint
        response = requests.get(
            url,
            params=params or {},
            timeout=timeout,
            headers={"User-Agent": "MATCHLAB/4.0 football research"},
        )
        response.raise_for_status()
        payload = response.json()
        return payload

    def search_team(self, name):
        data = self.get("/searchteams.php", {"t": name})
        return data.get("teams") or []

    def search_event(self, title):
        data = self.get("/searchevents.php", {"e": title})
        return data.get("event") or data.get("events") or []

    def events_day(self, day):
        data = self.get("/eventsday.php", {"d": day, "s": "Soccer"})
        return data.get("events") or []

    def eventslast(self, team_id):
        data = self.get("/eventslast.php", {"id": team_id})
        return data.get("results") or data.get("events") or []

    def eventsnext(self, team_id):
        data = self.get("/eventsnext.php", {"id": team_id})
        return data.get("events") or data.get("results") or []

    def event(self, event_id):
        data = self.get("/lookupevent.php", {"id": event_id})
        events = data.get("events") or []
        return events[0] if events else None

football = FreeFootballData()

# Automatically reconcile completed predictions with final scores when possible.
try:
    sync_completed_predictions()
except Exception:
    pass

def team_name_matches(actual, requested):
    a, r = norm(actual), norm(requested)
    if a == r:
        return True
    a2 = re.sub(r"[^a-z0-9 ]", "", a)
    r2 = re.sub(r"[^a-z0-9 ]", "", r)
    return a2 == r2 or r2 in a2 or a2 in r2

def choose_team(candidates, requested):
    if not candidates:
        return None
    exact = [x for x in candidates if team_name_matches(x.get("strTeam"), requested)]
    return exact[0] if exact else candidates[0]

def event_matches_fixture(event, home_name, away_name):
    h = event.get("strHomeTeam", "")
    a = event.get("strAwayTeam", "")
    return team_name_matches(h, home_name) and team_name_matches(a, away_name)

def event_to_fixture(event):
    return {
        "fixture": {
            "id": event.get("idEvent"),
            "date": f"{event.get('dateEvent','')}T{event.get('strTime','00:00:00')}",
            "venue": {"name": event.get("strVenue") or "Unknown venue"},
            "status": {"short": "FT" if event.get("intHomeScore") is not None else "NS"},
        },
        "league": {"name": event.get("strLeague") or "Unknown competition"},
        "teams": {
            "home": {"id": event.get("idHomeTeam"), "name": event.get("strHomeTeam")},
            "away": {"id": event.get("idAwayTeam"), "name": event.get("strAwayTeam")},
        },
        "goals": {
            "home": safe_float(event.get("intHomeScore")),
            "away": safe_float(event.get("intAwayScore")),
        },
        "raw_event": event,
    }

def _extract_iso_dates(text_value):
    """Extract common ISO, English, and Spanish calendar dates from web text."""
    s = str(text_value or "")
    found = []

    # ISO / slash forms.
    for y, m, d in re.findall(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", s):
        try:
            found.append(date(int(y), int(m), int(d)))
        except ValueError:
            pass

    month_map = {
        "jan": 1, "january": 1, "ene": 1, "enero": 1,
        "feb": 2, "february": 2, "febrero": 2,
        "mar": 3, "march": 3, "marzo": 3,
        "apr": 4, "april": 4, "abr": 4, "abril": 4,
        "may": 5, "mayo": 5,
        "jun": 6, "june": 6, "junio": 6,
        "jul": 7, "july": 7, "julio": 7,
        "aug": 8, "august": 8, "ago": 8, "agosto": 8,
        "sep": 9, "sept": 9, "september": 9, "septiembre": 9,
        "oct": 10, "october": 10, "octubre": 10,
        "nov": 11, "november": 11, "noviembre": 11,
        "dec": 12, "december": 12, "dic": 12, "diciembre": 12,
    }

    # "October 21, 2026" / "Oct 21 2026"
    for mon, day_num, year_num in re.findall(
        r"\b([A-Za-zÁÉÍÓÚáéíóú]{3,12})\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*|\s+)(20\d{2})\b",
        s,
    ):
        m = month_map.get(mon.lower())
        if m:
            try:
                found.append(date(int(year_num), m, int(day_num)))
            except ValueError:
                pass

    # "21 October 2026" / "21 Oct 2026"
    for day_num, mon, year_num in re.findall(
        r"\b(\d{1,2})\s+([A-Za-zÁÉÍÓÚáéíóú]{3,12})\s+(20\d{2})\b",
        s,
    ):
        m = month_map.get(mon.lower())
        if m:
            try:
                found.append(date(int(year_num), m, int(day_num)))
            except ValueError:
                pass

    # Spanish: "21 de octubre de 2026"
    for day_num, mon, year_num in re.findall(
        r"\b(\d{1,2})\s+de\s+([A-Za-zÁÉÍÓÚáéíóú]{3,12})\s+de\s+(20\d{2})\b",
        s,
        flags=re.IGNORECASE,
    ):
        m = month_map.get(mon.lower())
        if m:
            try:
                found.append(date(int(year_num), m, int(day_num)))
            except ValueError:
                pass

    # Remove duplicates while preserving order.
    unique = []
    seen = set()
    for d in found:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return unique

def _web_fixture_candidates(home_name, away_name, match_date, competition_hint=""):
    """Use public web/news search as a genuine fixture-discovery layer."""
    queries = [
        f'"{home_name}" "{away_name}" football {match_date.year}',
        f'"{home_name}" "{away_name}" fixture {competition_hint}'.strip(),
        f'"{home_name}" vs "{away_name}" {competition_hint}'.strip(),
    ]
    results = []
    seen = set()

    for q in queries:
        try:
            found = public_web_search(q, max_results=8)
        except Exception:
            found = []
        for r in found:
            key = r.get("url") or r.get("title")
            if key and key not in seen:
                seen.add(key)
                results.append(r)

    candidates = []
    hn = norm(home_name)
    an = norm(away_name)

    for r in results:
        text = f"{r.get('title','')} {r.get('content','')}"
        blob = norm(text)

        # Search engines often abbreviate club names (e.g. "Man Utd") or
        # omit punctuation. Accept a strong token overlap instead of requiring
        # the entire normalized name to appear verbatim.
        def name_present(name):
            n = norm(name)
            if n in blob:
                return True
            tokens = [t for t in re.findall(r"[a-z0-9]+", n) if len(t) >= 3]
            if not tokens:
                return False
            hits = sum(1 for t in tokens if t in blob)
            return hits >= max(1, min(2, len(tokens)))

        if not name_present(home_name) or not name_present(away_name):
            continue

        dates_found = _extract_iso_dates(text)
        candidates.append({
            "result": r,
            "dates": dates_found,
        })

    return candidates


def find_fixture(home_name, away_name, match_date, competition_hint=""):
    wanted_date = match_date.isoformat()

    # ---------- 1. Free structured football source ----------
    titles = [
        f"{home_name} vs {away_name}",
        f"{home_name} v {away_name}",
        f"{home_name} {away_name}",
    ]
    for title in titles:
        try:
            events = football.search_event(title)
            for event in events:
                if (
                    str(event.get("dateEvent", ""))[:10] == wanted_date
                    and event_matches_fixture(event, home_name, away_name)
                ):
                    fixture = event_to_fixture(event)
                    return fixture, fixture["teams"]["home"], fixture["teams"]["away"]
        except Exception:
            continue

    try:
        events = football.events_day(wanted_date)
        for event in events:
            if event_matches_fixture(event, home_name, away_name):
                fixture = event_to_fixture(event)
                return fixture, fixture["teams"]["home"], fixture["teams"]["away"]
    except Exception:
        pass

    # ---------- 1B. Team-schedule fallback ----------
    # Some free-data searches miss a fixture even though it is present in
    # the team's upcoming schedule. Search both teams' schedules before
    # relying on HTML/news search. This is more reliable for ordinary
    # fixtures and does not require a paid API key.
    try:
        home_candidates = football.search_team(home_name)
        away_candidates = football.search_team(away_name)
        home_match = choose_team(home_candidates, home_name)
        away_match = choose_team(away_candidates, away_name)
        home_id = home_match.get("idTeam") if home_match else None
        away_id = away_match.get("idTeam") if away_match else None

        schedule_events = []
        for team_id in (home_id, away_id):
            if not team_id:
                continue
            try:
                schedule_events.extend(football.eventsnext(team_id))
            except Exception:
                continue

        seen_events = set()
        for event in schedule_events:
            eid = str(event.get("idEvent") or "")
            if eid and eid in seen_events:
                continue
            if eid:
                seen_events.add(eid)
            event_date = str(event.get("dateEvent") or "")[:10]
            if event_date == wanted_date and event_matches_fixture(event, home_name, away_name):
                fixture = event_to_fixture(event)
                return fixture, fixture["teams"]["home"], fixture["teams"]["away"]
    except Exception:
        pass

    # ---------- 2. Public Internet fixture discovery ----------
    try:
        candidates = _web_fixture_candidates(
            home_name, away_name, match_date, competition_hint
        )

        # Prefer an exact date, then a one-day timezone/display difference.
        best = None
        best_distance = 999

        for c in candidates:
            for d in c["dates"]:
                distance = abs((d - match_date).days)
                if distance < best_distance:
                    best_distance = distance
                    best = (c, d)

        if best:
            chosen, verified_date = best
            best_distance = abs((verified_date - match_date).days)

            accepted = best_distance <= 1
            if best_distance == 2:
                corroborating = 0
                for other in candidates:
                    if other is chosen:
                        continue
                    if any(abs((d - verified_date).days) <= 1 for d in other["dates"]):
                        corroborating += 1
                accepted = corroborating >= 1

            if accepted:
                chosen_result = chosen["result"]
                home_candidates = football.search_team(home_name)
                away_candidates = football.search_team(away_name)
                home_match = choose_team(home_candidates, home_name)
                away_match = choose_team(away_candidates, away_name)

                home_id = home_match.get("idTeam") if home_match else None
                away_id = away_match.get("idTeam") if away_match else None

                fixture = {
                    "fixture": {
                        "id": f"WEB-{verified_date.isoformat()}-{norm(home_name)}-{norm(away_name)}",
                        "date": f"{verified_date.isoformat()}T00:00:00",
                        "venue": {"name": "Web-researched venue / verify source"},
                        "status": {"short": "NS"},
                    },
                    "league": {
                        "name": competition_hint or "Web-researched competition"
                    },
                    "teams": {
                        "home": {"id": home_id, "name": home_name},
                        "away": {"id": away_id, "name": away_name},
                    },
                    "goals": {"home": None, "away": None},
                    "raw_event": {
                        "web_verified": True,
                        "requested_date": wanted_date,
                        "verified_date": verified_date.isoformat(),
                        "date_difference_days": best_distance,
                        "fixture_source": chosen_result,
                        "research_results": [x["result"] for x in candidates],
                    },
                }
                return fixture, fixture["teams"]["home"], fixture["teams"]["away"]

    except Exception:
        pass

    # ---------- 3. Last-resort user-supplied fixture fallback ----------
    # Do not block the prediction engine just because a free provider/search
    # engine is temporarily unavailable. The user has explicitly supplied the
    # two teams and date, so MATCHLAB can proceed as a PROVISIONAL fixture and
    # clearly label it. Current team news and any available form data are still
    # collected separately below. This keeps the prediction engine usable even
    # when free fixture databases rate-limit or search pages change.
    try:
        home_candidates = football.search_team(home_name)
    except Exception:
        home_candidates = []
    try:
        away_candidates = football.search_team(away_name)
    except Exception:
        away_candidates = []

    home_match = choose_team(home_candidates, home_name)
    away_match = choose_team(away_candidates, away_name)

    home_id = home_match.get("idTeam") if home_match else None
    away_id = away_match.get("idTeam") if away_match else None
    canonical_home = home_match.get("strTeam") if home_match else home_name
    canonical_away = away_match.get("strTeam") if away_match else away_name

    fixture = {
        "fixture": {
            "id": f"PROVISIONAL-{wanted_date}-{norm(home_name)}-{norm(away_name)}",
            "date": f"{wanted_date}T00:00:00",
            "venue": {"name": "Venue not independently verified"},
            "status": {"short": "NS"},
        },
        "league": {"name": competition_hint or "Competition not independently verified"},
        "teams": {
            "home": {"id": home_id, "name": canonical_home},
            "away": {"id": away_id, "name": canonical_away},
        },
        "goals": {"home": None, "away": None},
        "raw_event": {
            "provisional": True,
            "requested_date": wanted_date,
            "fixture_source": "User-supplied fixture; external verification unavailable",
            "verification_note": (
                "Free fixture sources did not return a matching event. "
                "MATCHLAB proceeded without inventing a different date or result."
            ),
        },
    }
    return fixture, fixture["teams"]["home"], fixture["teams"]["away"]

def event_to_row(event, team_id):
    if not event:
        return None
    hg = safe_float(event.get("intHomeScore"))
    ag = safe_float(event.get("intAwayScore"))
    if hg is None or ag is None:
        return None

    home_id = str(event.get("idHomeTeam", ""))
    away_id = str(event.get("idAwayTeam", ""))
    tid = str(team_id)

    if home_id == tid:
        return {
            "date": event.get("dateEvent", ""),
            "team_id": team_id,
            "opponent_id": event.get("idAwayTeam"),
            "venue": "home",
            "gf": hg, "ga": ag,
            "result": "W" if hg > ag else "D" if hg == ag else "L",
            "competition": event.get("strLeague", ""),
        }
    if away_id == tid:
        return {
            "date": event.get("dateEvent", ""),
            "team_id": team_id,
            "opponent_id": event.get("idHomeTeam"),
            "venue": "away",
            "gf": ag, "ga": hg,
            "result": "W" if ag > hg else "D" if ag == hg else "L",
            "competition": event.get("strLeague", ""),
        }
    return None

@st.cache_data(ttl=900, show_spinner=False)
def collect_recent_events(team_id, days=180):
    rows = []
    seen = set()
    today = date.today()
    for offset in range(1, days + 1):
        day = (today - timedelta(days=offset)).isoformat()
        try:
            events = football.events_day(day)
        except Exception:
            continue
        for event in events:
            eid = str(event.get("idEvent", ""))
            if not eid or eid in seen:
                continue
            row = event_to_row(event, team_id)
            if row:
                seen.add(eid)
                rows.append(row)
        if len(rows) >= 25:
            break
        time.sleep(0.03)
    rows.sort(key=lambda x: x.get("date") or "")
    return rows[-25:]

@st.cache_data(ttl=900, show_spinner=False)
def collect_h2h_from_web(home_name, away_name):
    return {
        "provider": "Public Internet search",
        "results": public_web_search(
            f"{home_name} {away_name} head to head previous meetings results",
            max_results=8,
        ),
    }

# ============================================================
# FREE INTERNET SEARCH — NO SEARCH API KEY
# ============================================================

class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = False
    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self.skip = True
    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"}:
            self.skip = False
    def handle_data(self, data):
        if not self.skip:
            text = " ".join(data.split())
            if text:
                self.parts.append(text)
    def text(self):
        return " ".join(self.parts)

def parse_ddg_results(page):
    results = []
    # DDG HTML commonly uses result__a and result__snippet.
    links = re.findall(
        r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        page,
        flags=re.I | re.S,
    )
    snippets = re.findall(
        r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>',
        page,
        flags=re.I | re.S,
    )
    for i, (url, title_html) in enumerate(links):
        title = re.sub("<.*?>", " ", title_html)
        title = html.unescape(" ".join(title.split()))
        snippet = ""
        if i < len(snippets):
            snippet = re.sub("<.*?>", " ", snippets[i])
            snippet = html.unescape(" ".join(snippet.split()))
        if url.startswith("//duckduckgo.com/l/?"):
            m = re.search(r"uddg=([^&]+)", url)
            if m:
                from urllib.parse import unquote
                url = unquote(m.group(1))
        results.append({"title": title, "url": url, "content": snippet})
    return results

def ddg_search(query, max_results=5):
    url = "https://html.duckduckgo.com/html/"
    response = requests.get(
        url,
        params={"q": query},
        timeout=12,
        headers={"User-Agent": "Mozilla/5.0 MATCHLAB research"},
    )
    response.raise_for_status()
    return parse_ddg_results(response.text)[:max_results]

def google_news_search(query, max_results=5):
    url = "https://news.google.com/rss/search"
    response = requests.get(
        url,
        params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
        timeout=12,
        headers={"User-Agent": "Mozilla/5.0 MATCHLAB research"},
    )
    response.raise_for_status()
    import xml.etree.ElementTree as ET
    root = ET.fromstring(response.text)
    out = []
    for item in root.findall(".//item")[:max_results]:
        title = item.findtext("title") or ""
        link = item.findtext("link") or ""
        desc = item.findtext("description") or ""
        desc = re.sub("<.*?>", " ", desc)
        out.append({
            "title": html.unescape(" ".join(title.split())),
            "url": link,
            "content": html.unescape(" ".join(desc.split())),
        })
    return out

@st.cache_data(ttl=600, show_spinner=False)
def public_web_search(query, max_results=5):
    # Query both public sources before truncating. The previous implementation
    # returned as soon as the first source filled the quota, so Google News was
    # effectively ignored whenever DuckDuckGo returned enough rows.
    results = []
    seen = set()
    per_source = max(2, max_results)
    for searcher in (ddg_search, google_news_search):
        try:
            found = searcher(query, max_results=per_source)
        except Exception:
            found = []
        for item in found:
            key = item.get("url") or item.get("title")
            if key and key not in seen:
                seen.add(key)
                results.append(item)
    return results[:max_results]

@st.cache_data(ttl=600, show_spinner=False)
def live_question_web_research(home_name, away_name, match_date, competition_hint, question):
    """Fresh public-web research for an interactive MATCHLAB question."""
    q=norm(question)
    base=f'"{home_name}" "{away_name}" {competition_hint} {match_date}'
    queries=[]
    if any(k in q for k in ["injur","lineup","suspension","available","missing","news","squad"]):
        queries += [
            f'{base} latest team news injuries suspensions lineup',
            f'"{home_name}" latest team news injury lineup {match_date}',
            f'"{away_name}" latest team news injury lineup {match_date}',
        ]
    if any(k in q for k in ["form","recent","last five","home","away","attack","defen"]):
        queries += [
            f'{base} recent form last 5 results home away goals',
            f'"{home_name}" recent results form goals {match_date}',
            f'"{away_name}" recent results form goals {match_date}',
        ]
    if any(k in q for k in ["why","prediction","winner","win","draw","result","risk","wrong","bet"]):
        queries += [f'{base} prediction preview analysis', f'{base} preview prediction team news', f'{base} expected lineup preview latest']
    if any(k in q for k in ["weather","rain","wind","pitch","condition"]):
        queries += [f'{base} weather stadium conditions {match_date}']
    if any(k in q for k in ["motivation","table","position","relegation","promotion","title","europe"]):
        queries += [f'{base} league table motivation promotion relegation {match_date}']
    queries += [f'{base} {question}', f'"{home_name}" "{away_name}" {question}']
    dedup=[]; seen_q=set()
    for item in queries:
        key=norm(item)
        if key not in seen_q:
            seen_q.add(key); dedup.append(item)
    all_results=[]; seen_urls=set()
    for search_q in dedup[:8]:
        try: found=public_web_search(search_q,max_results=5)
        except Exception: found=[]
        for r in found:
            key=r.get("url") or r.get("title") or ""
            if key in seen_urls: continue
            seen_urls.add(key)
            item=dict(r); item["query"]=search_q; all_results.append(item)
            if len(all_results)>=30: break
        if len(all_results)>=30: break
    return {"provider":"Fresh public Internet research (DuckDuckGo + Google News RSS)","question":question,"queries":dedup[:8],"results":all_results,"searched_at":datetime.utcnow().isoformat(timespec="seconds")+"Z"}


def web_question_context(live_research, max_chars=7000):
    chunks=[]
    for r in (live_research or {}).get("results",[])[:24]:
        chunks.append(f"{r.get('title','')} | {r.get('content','')} | {r.get('url','')}")
    return "\n".join(chunks)[:max_chars]


@st.cache_data(ttl=600, show_spinner=False)
def public_web_research(home_name, away_name, match_date, competition_hint=""):
    queries = [
        f'"{home_name}" "{away_name}" {match_date} football',
        f'"{home_name}" latest news injuries lineup',
        f'"{away_name}" latest news injuries lineup',
        f'"{home_name}" recent results form goals',
        f'"{away_name}" recent results form goals',
        f'"{home_name}" "{away_name}" head to head',
    ]
    all_results = []
    seen = set()
    for q in queries:
        try:
            found = public_web_search(q, max_results=4)
        except Exception:
            found = []
        for r in found:
            key = r.get("url") or r.get("title")
            if key in seen:
                continue
            seen.add(key)
            r["query"] = q
            all_results.append(r)
            if len(all_results) >= 20:
                break
        if len(all_results) >= 20:
            break

    return {
        "provider": "Public Internet search (DuckDuckGo + Google News RSS)",
        "results": all_results,
        "queries": queries,
    }

def research_text(web_research, max_chars=9000):
    chunks = []
    for r in web_research.get("results", []):
        chunks.append(
            f"{r.get('title','')} | {r.get('content','')} | {r.get('url','')}"
        )
    return "\n".join(chunks)[:max_chars]

def web_injury_items(web_research, team_name):
    """Conservative extraction: only label as a reported issue when search text contains clear keywords."""
    keywords = ("injury", "injured", "out", "suspended", "suspension", "doubt", "unavailable", "miss")
    team = norm(team_name)
    items = []
    for r in web_research.get("results", []):
        text = norm(f"{r.get('title','')} {r.get('content','')}")
        if team in text and any(k in text for k in keywords):
            items.append({
                "player": "Reported availability issue",
                "type": "Web report",
                "reason": r.get("title") or r.get("content", "")[:180],
                "source": r.get("url", ""),
            })
    return items[:6]

# ============================================================
# WEB FORM PARSING
# ============================================================

def profile_from_web_if_needed(profile, web_research, team_name):
    if profile["matches"] >= 3:
        return profile

    # Do not fabricate scores from prose. Instead add a small confidence penalty
    # later and use conservative league-neutral priors.
    return profile


# ============================================================
# ADVANCED TEAM NEWS + FIXTURE CONGESTION ENGINE
# ============================================================

ROTATION_TERMS = (
    "rotate", "rotation", "rest players", "rested", "resting", "rotated",
    "changes expected", "weakened side", "squad rotation", "manage minutes",
    "minutes restriction", "rest key", "key players rested",
)
BIG_MATCH_TERMS = (
    "champions league", "europa league", "conference league", "fa cup",
    "carabao cup", "copa del rey", "coppa italia", "dfb pokal", "cup final",
    "semi-final", "semifinal", "quarter-final", "quarterfinal", "final",
    "derby", "playoff", "play-off", "promotion final", "relegation six-pointer",
)
ABSENCE_TERMS = (
    "injury", "injured", "suspended", "suspension", "unavailable", "doubtful",
    "misses", "miss", "ruled out", "out of", "fitness test", "illness",
)
LINEUP_TERMS = (
    "predicted lineup", "predicted line-up", "starting xi", "starting eleven",
    "confirmed lineup", "confirmed line-up", "team news", "likely lineup",
)


def _source_quality_score(result):
    url = norm(result.get("url", ""))
    title = norm(result.get("title", ""))
    score = 0
    if any(x in url for x in (".gov", "uefa.com", "fifa.com", "premierleague.com")):
        score += 5
    if any(x in url for x in ("bbc.", "skysports.com", "espn.com", "theguardian.com", "reuters.com", "apnews.com")):
        score += 4
    if any(x in title for x in ("official", "confirmed", "press conference")):
        score += 2
    return score


@st.cache_data(ttl=600, show_spinner=False)
def research_team_news(home_team, away_team, match_date, competition_hint=""):
    """Targeted team-news research. Facts stay evidence-backed; weak snippets never become fake player data."""
    queries = [
        f'"{home_team}" team news injuries suspensions {match_date}',
        f'"{away_team}" team news injuries suspensions {match_date}',
        f'"{home_team}" predicted lineup starting XI {match_date}',
        f'"{away_team}" predicted lineup starting XI {match_date}',
        f'"{home_team}" rotation rest players next match',
        f'"{away_team}" rotation rest players next match',
        f'"{home_team}" manager press conference team news',
        f'"{away_team}" manager press conference team news',
    ]
    results=[]; seen=set()
    for q in queries:
        try:
            found=public_web_search(q,max_results=6)
        except Exception:
            found=[]
        for r in found:
            key=r.get("url") or r.get("title")
            if not key or key in seen:
                continue
            seen.add(key)
            item=dict(r); item["query"]=q; item["quality"]=_source_quality_score(item)
            results.append(item)
    results.sort(key=lambda x:(x.get("quality",0), len(x.get("content", ""))), reverse=True)

    def team_pack(team):
        tn=norm(team); facts=[]; rotation=[]; lineup=[]; absences=[]
        for r in results:
            text=norm(f'{r.get("title","")} {r.get("content","")}')
            if tn not in text:
                continue
            if any(k in text for k in ABSENCE_TERMS):
                absences.append(r)
            if any(k in text for k in ROTATION_TERMS):
                rotation.append(r)
            if any(k in text for k in LINEUP_TERMS):
                lineup.append(r)
            if any(k in text for k in ABSENCE_TERMS+ROTATION_TERMS+LINEUP_TERMS):
                facts.append({"Topic":r.get("query",""),"Source":r.get("title",""),"Evidence":r.get("content","")[:450],"URL":r.get("url","")})
        return {
            "facts":facts[:10],
            "absence_sources":absences[:6],
            "rotation_sources":rotation[:6],
            "lineup_sources":lineup[:6],
            "evidence_count":len(facts),
        }
    return {"home":team_pack(home_team),"away":team_pack(away_team),"results":results[:30],"queries":queries}


def _event_date(event):
    raw=str(event.get("dateEvent") or event.get("date") or "")[:10]
    try:
        return date.fromisoformat(raw)
    except Exception:
        return None


def _fixture_label(event):
    return f"{event.get('strHomeTeam','?')} vs {event.get('strAwayTeam','?')}"


def _competition_importance(name):
    text=norm(name)
    if any(k in text for k in ("champions league", "europa league", "conference league")):
        return 5
    if any(k in text for k in ("final", "semi-final", "semifinal", "quarter-final", "quarterfinal", "playoff", "play-off", "derby")):
        return 5
    if any(k in text for k in ("cup", "copa", "coppa", "pokal")):
        return 3
    return 1


@st.cache_data(ttl=900, show_spinner=False)
def future_fixture_research(team_name, team_id, after_date, competition_hint="", days=21):
    """Find the team's next fixtures and flag congestion / likely rotation spots."""
    events=[]
    if team_id:
        try:
            events.extend(football.eventsnext(team_id) or [])
        except Exception:
            pass
    clean=[]; seen=set()
    for e in events:
        eid=str(e.get("idEvent") or _fixture_label(e))
        if eid in seen: continue
        seen.add(eid)
        d=_event_date(e)
        if not d or d <= after_date or d > after_date+timedelta(days=days):
            continue
        clean.append(e)
    clean.sort(key=lambda e:_event_date(e) or date.max)

    # Web fallback catches competitions missing from the free schedule endpoint.
    if len(clean)<3:
        queries=[
            f'"{team_name}" fixtures next matches {after_date.isoformat()}',
            f'"{team_name}" upcoming fixtures next 3 weeks',
        ]
        for q in queries:
            try: found=public_web_search(q,max_results=8)
            except Exception: found=[]
            for r in found:
                text=f"{r.get('title','')} {r.get('content','')}"
                dates=_extract_iso_dates(text)
                for d in dates:
                    if after_date < d <= after_date+timedelta(days=days):
                        clean.append({"strHomeTeam":team_name,"strAwayTeam":"Upcoming opponent (web)","dateEvent":d.isoformat(),"strLeague":"Web schedule","strEvent":"Web-researched fixture","_web_source":r})
                        break
            if len(clean)>=3: break

    dedup=[]; seen_dates=set()
    for e in sorted(clean,key=lambda e:_event_date(e) or date.max):
        d=_event_date(e)
        key=(d, norm(e.get("strHomeTeam")), norm(e.get("strAwayTeam")))
        if key in seen_dates: continue
        seen_dates.add(key); dedup.append(e)

    rows=[]
    for e in dedup[:8]:
        d=_event_date(e)
        gap=(d-after_date).days if d else None
        comp=e.get("strLeague") or "Unknown competition"
        rows.append({
            "date":d.isoformat() if d else "",
            "fixture":_fixture_label(e),
            "competition":comp,
            "days_after_target":gap,
            "importance":_competition_importance(comp),
            "source":("web" if e.get("_web_source") else "TheSportsDB"),
            "url":(e.get("_web_source") or {}).get("url","") if e.get("_web_source") else "",
        })

    next_big=next((r for r in rows if r["importance"]>=3),None)
    return {"team":team_name,"fixtures":rows,"next_big":next_big}


def schedule_adjustment(schedule, news_pack):
    """Small, capped rotation/fatigue effect. Schedule context cannot overpower core team strength."""
    fixtures=schedule.get("fixtures",[])
    if not fixtures:
        return 1.0, "No reliable upcoming-fixture congestion signal."
    near=[f for f in fixtures if f.get("days_after_target") is not None and f["days_after_target"]<=4]
    big=[f for f in fixtures if f.get("importance",1)>=3 and f.get("days_after_target",99)<=7]
    text=" ".join(norm(f"{x.get('title','')} {x.get('content','')}" ) for x in news_pack.get("rotation_sources",[]))
    explicit=any(k in text for k in ROTATION_TERMS)
    if explicit and big:
        return 0.972, f"Explicit rotation/rest reporting plus a major fixture within 7 days ({big[0]['fixture']})."
    if near and big:
        return 0.982, f"Compressed schedule with a major upcoming fixture ({big[0]['fixture']})."
    if near:
        return 0.990, "Short turnaround detected; a small fatigue adjustment is applied."
    return 1.0, "No meaningful fixture-congestion penalty applied."


def model_uncertainty(coverage, profiles, matrix_result, provisional=False, news_count=0):
    gap=sorted([matrix_result["home_win"],matrix_result["draw"],matrix_result["away_win"]],reverse=True)
    spread=gap[0]-gap[1]
    u=0.24
    u -= max(0,coverage-50)*0.0015
    u -= min(profiles[0].get("matches",0),profiles[1].get("matches",0))*0.004
    u -= min(news_count,10)*0.002
    u += max(0,0.08-spread)*0.25
    if provisional: u += 0.08
    return clamp(u,0.05,0.35)

# ============================================================
# LIVE STADIUM WEATHER + MOTIVATION RESEARCH
# ============================================================

WEATHER_CODE_LABELS = {
    0: "Clear", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Freezing fog", 51: "Light drizzle", 53: "Drizzle",
    55: "Heavy drizzle", 61: "Light rain", 63: "Rain", 65: "Heavy rain",
    66: "Freezing rain", 67: "Heavy freezing rain", 71: "Light snow",
    73: "Snow", 75: "Heavy snow", 77: "Snow grains", 80: "Rain showers",
    81: "Rain showers", 82: "Heavy rain showers", 85: "Snow showers",
    86: "Heavy snow showers", 95: "Thunderstorm", 96: "Thunderstorm + hail",
    99: "Severe thunderstorm + hail",
}

def _geocode_stadium(venue_name, home_team):
    """Resolve the host stadium to coordinates using the free Nominatim service."""
    candidates = []
    if venue_name and "unknown" not in norm(venue_name) and "not independently" not in norm(venue_name):
        candidates.extend([f"{venue_name}, {home_team}", venue_name])
    candidates.extend([f"{home_team} stadium", home_team])
    for query in candidates:
        try:
            response = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": query, "format": "jsonv2", "limit": 3},
                timeout=10,
                headers={"User-Agent": "MATCHLAB-Live/1.0 (football prediction research)"},
            )
            response.raise_for_status()
            rows = response.json() or []
            if rows:
                row = rows[0]
                return {
                    "latitude": float(row["lat"]),
                    "longitude": float(row["lon"]),
                    "display_name": row.get("display_name", query),
                    "query": query,
                }
        except Exception:
            continue
    return None

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_stadium_weather(venue_name, home_team, fixture_datetime, match_date):
    """Fetch current/future host-stadium weather from Open-Meteo. No API key."""
    geo = _geocode_stadium(venue_name, home_team)
    if not geo:
        return {"available": False, "reason": "Host stadium could not be geocoded."}

    target = match_date.isoformat()
    today = date.today()
    days_ahead = (match_date - today).days
    if days_ahead < 0:
        endpoint = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": geo["latitude"], "longitude": geo["longitude"],
            "start_date": target, "end_date": target, "daily": "weathercode,temperature_2m_mean,precipitation_sum,windspeed_10m_max",
            "timezone": "auto",
        }
        mode = "historical"
    else:
        endpoint = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": geo["latitude"], "longitude": geo["longitude"],
            "start_date": target, "end_date": target,
            "daily": "weathercode,temperature_2m_mean,precipitation_sum,windspeed_10m_max",
            "timezone": "auto",
            "forecast_days": 16,
        }
        mode = "forecast"
    try:
        response = requests.get(endpoint, params=params, timeout=12, headers={"User-Agent": "MATCHLAB-Live/1.0"})
        response.raise_for_status()
        data = response.json()
        daily = data.get("daily") or {}
        if not daily.get("time"):
            return {"available": False, "reason": "Weather provider returned no data for this match date.", "geo": geo}
        code = int((daily.get("weathercode") or [0])[0])
        temp = safe_float((daily.get("temperature_2m_mean") or [None])[0])
        precip = safe_float((daily.get("precipitation_sum") or [None])[0], 0.0)
        wind = safe_float((daily.get("windspeed_10m_max") or [None])[0], 0.0)
        label = WEATHER_CODE_LABELS.get(code, "Variable conditions")
        return {
            "available": True, "mode": mode, "label": label, "weather_code": code,
            "temperature_c": temp, "precipitation_mm": precip, "wind_kmh": wind,
            "geo": geo, "source": "Open-Meteo + OpenStreetMap Nominatim",
            "environment_factor": weather_environment_factor(label, precip, wind, temp),
        }
    except Exception as exc:
        return {"available": False, "reason": f"Weather service unavailable: {type(exc).__name__}", "geo": geo}

def weather_environment_factor(label, precipitation_mm, wind_kmh, temperature_c):
    """Restrained total-goals adjustment; extreme conditions reduce scoring, not certainty."""
    factor = 1.0
    if precipitation_mm >= 15: factor *= 0.94
    elif precipitation_mm >= 6: factor *= 0.975
    if wind_kmh >= 45: factor *= 0.92
    elif wind_kmh >= 30: factor *= 0.96
    if temperature_c is not None:
        if temperature_c >= 33: factor *= 0.965
        elif temperature_c <= 2: factor *= 0.975
    if "snow" in label.lower() or "thunderstorm" in label.lower(): factor *= 0.965
    return clamp(factor, 0.86, 1.02)

MOTIVATION_POSITIVE = ("must win", "need to win", "win to", "title race", "promotion", "qualif", "survival", "avoid relegation", "relegation battle", "final", "semi-final", "semifinal", "decider", "second leg")
MOTIVATION_NEGATIVE = ("nothing to play for", "dead rubber", "already eliminated", "already relegated", "rest players", "rotate", "rotation", "rotating", "rested")

def research_motivation(home_team, away_team, competition_hint, web_research):
    """Score motivation only from explicit public evidence; weak/no evidence stays neutral."""
    queries = [
        f'"{home_team}" motivation must win {competition_hint}',
        f'"{away_team}" motivation must win {competition_hint}',
        f'"{home_team}" rotation rest players {competition_hint}',
        f'"{away_team}" rotation rest players {competition_hint}',
        f'"{home_team}" "{away_team}" stakes qualification relegation title',
    ]
    results = []
    seen = set()
    for q in queries:
        try:
            found = public_web_search(q, max_results=4)
        except Exception:
            found = []
        for r in found:
            key = r.get("url") or r.get("title")
            if key and key not in seen:
                seen.add(key); r = dict(r); r["query"] = q; results.append(r)
    def team_score(team):
        score = 0; evidence = []
        tn = norm(team)
        for r in results:
            text = norm(f"{r.get('title','')} {r.get('content','')}")
            if tn not in text:
                continue
            pos = [p for p in MOTIVATION_POSITIVE if p in text]
            neg = [n for n in MOTIVATION_NEGATIVE if n in text]
            if pos or neg:
                score += min(2, len(pos)) - min(2, len(neg))
                evidence.append({"Source": r.get("title", ""), "Signal": ", ".join(pos + neg), "Evidence": r.get("content", "")[:350], "URL": r.get("url", "")})
        level = "High" if score >= 2 else "Low" if score <= -2 else "Normal/unclear"
        return {"score": clamp(score, -3, 3), "level": level, "evidence": evidence[:5]}
    return {"home": team_score(home_team), "away": team_score(away_team), "results": results[:15], "source": "Public Internet research; neutral when evidence is weak"}

def motivation_adjustments(motivation):
    hs = motivation.get("home", {}).get("score", 0)
    aw = motivation.get("away", {}).get("score", 0)
    # Small effect only: motivation should not overpower actual football strength.
    return clamp(1.0 + hs * 0.012, 0.965, 1.036), clamp(1.0 + aw * 0.012, 0.965, 1.036)


# ============================================================
# HISTORICAL / RECENT TEAM MODEL
# ============================================================

def completed_fixture(fixture):
    status = fixture.get("fixture", {}).get("status", {}).get("short")
    return status in {"FT", "AET", "PEN"}


def extract_match_row(fixture, team_id):
    if not completed_fixture(fixture):
        return None

    home = fixture.get("teams", {}).get("home", {})
    away = fixture.get("teams", {}).get("away", {})
    goals = fixture.get("goals", {})

    hg = goals.get("home")
    ag = goals.get("away")

    if hg is None or ag is None:
        return None

    if home.get("id") == team_id:
        return {
            "date": fixture.get("fixture", {}).get("date"),
            "team_id": team_id,
            "opponent_id": away.get("id"),
            "venue": "home",
            "gf": float(hg),
            "ga": float(ag),
            "result": "W" if hg > ag else "D" if hg == ag else "L",
            "competition": fixture.get("league", {}).get("name", ""),
        }

    if away.get("id") == team_id:
        return {
            "date": fixture.get("fixture", {}).get("date"),
            "team_id": team_id,
            "opponent_id": home.get("id"),
            "venue": "away",
            "gf": float(ag),
            "ga": float(hg),
            "result": "W" if ag > hg else "D" if ag == hg else "L",
            "competition": fixture.get("league", {}).get("name", ""),
        }

    return None


def weighted_average(values, weights):
    if not values:
        return None
    total_weight = sum(weights)
    if total_weight <= 0:
        return sum(values) / len(values)
    return sum(v * w for v, w in zip(values, weights)) / total_weight


def build_recent_team_profile(fixtures, team_id):
    rows = []

    for fixture in fixtures:
        row = extract_match_row(fixture, team_id)
        if row:
            rows.append(row)

    rows.sort(key=lambda x: x.get("date") or "")
    rows = rows[-25:]

    if not rows:
        return {
            "matches": 0,
            "home_matches": 0,
            "away_matches": 0,
            "gf": 1.35,
            "ga": 1.35,
            "home_gf": 1.45,
            "home_ga": 1.20,
            "away_gf": 1.25,
            "away_ga": 1.35,
            "points": 1.35,
            "recent_results": [],
            "last_match_date": None,
            "recent_gf5": 1.35,
            "recent_ga5": 1.35,
        }

    weights = []
    n = len(rows)
    for index in range(n):
        weights.append(0.82 ** (n - 1 - index))

    gf = weighted_average([r["gf"] for r in rows], weights)
    ga = weighted_average([r["ga"] for r in rows], weights)
    points = weighted_average(
        [3 if r["result"] == "W" else 1 if r["result"] == "D" else 0 for r in rows],
        weights,
    )

    home_rows = [r for r in rows if r["venue"] == "home"]
    away_rows = [r for r in rows if r["venue"] == "away"]

    def venue_metric(venue_rows, key, fallback):
        if not venue_rows:
            return fallback
        venue_weights = []
        m = len(venue_rows)
        for index in range(m):
            venue_weights.append(0.85 ** (m - 1 - index))
        return weighted_average(
            [r[key] for r in venue_rows],
            venue_weights,
        )

    return {
        "matches": len(rows),
        "home_matches": len(home_rows),
        "away_matches": len(away_rows),
        "gf": gf,
        "ga": ga,
        "home_gf": venue_metric(home_rows, "gf", 1.45),
        "home_ga": venue_metric(home_rows, "ga", 1.20),
        "away_gf": venue_metric(away_rows, "gf", 1.25),
        "away_ga": venue_metric(away_rows, "ga", 1.35),
        "points": points,
        "recent_results": [r["result"] for r in rows[-5:]],
        "last_match_date": rows[-1].get("date") if rows else None,
        "recent_gf5": weighted_average([r["gf"] for r in rows[-5:]], [0.88**(len(rows[-5:])-1-i) for i in range(len(rows[-5:]))]) if rows else gf,
        "recent_ga5": weighted_average([r["ga"] for r in rows[-5:]], [0.88**(len(rows[-5:])-1-i) for i in range(len(rows[-5:]))]) if rows else ga,
        "rows": rows,
    }


def profile_strength(profile):
    # Convert recent scoring/conceding into restrained multipliers.
    attack = clamp(
        1.0
        + (profile["gf"] - 1.35) * 0.09
        + (profile["points"] - 1.35) * 0.025,
        0.82,
        1.18,
    )

    defense = clamp(
        1.0 + (profile["ga"] - 1.35) * 0.075,
        0.84,
        1.16,
    )

    return attack, defense

# ============================================================
# CONTEXT ADJUSTMENTS
# ============================================================

def calculated_rest_days(profile, target_date):
    try:
        last=profile.get("last_match_date")
        if not last: return 5
        return max(0,(target_date-date.fromisoformat(str(last)[:10])).days)
    except Exception:
        return 5



FORMATION = {
    "4-3-3": (1.015, 1.000),
    "4-2-3-1": (1.008, 0.995),
    "4-4-2": (1.000, 1.000),
    "3-4-3": (1.018, 1.012),
    "3-5-2": (1.005, 0.990),
    "5-3-2": (0.985, 0.975),
    "5-4-1": (0.970, 0.965),
    "4-5-1": (0.985, 0.975),
}

MOTIVATION = {
    "Very Low": 0.975,
    "Low": 0.988,
    "Normal": 1.000,
    "High": 1.012,
    "Very High": 1.025,
}

WEATHER = {
    "Clear": 1.000,
    "Cloudy": 1.000,
    "Light Rain": 0.985,
    "Heavy Rain": 0.950,
    "Strong Wind": 0.940,
    "Extreme Heat": 0.965,
    "Snow": 0.930,
}

PITCH = {
    "Excellent": 1.000,
    "Good": 1.000,
    "Average": 0.985,
    "Poor": 0.960,
}


def form_adjustments(profile):
    attack, defense = profile_strength(profile)
    return attack, defense


def context_multiplier(
    formation,
    motivation,
    absences,
    absence_type,
    absence_importance,
    profile,
    rest_days,
    aggregate_for,
    aggregate_against,
    second_leg,
):
    attack, defense = FORMATION.get(
        formation,
        (1.0, 1.0),
    )

    attack *= MOTIVATION.get(
        motivation,
        1.0,
    )

    if profile:
        form_attack, form_defense = form_adjustments(profile)
        attack *= 1 + (form_attack - 1) * 0.70
        defense *= 1 + (form_defense - 1) * 0.70

    importance = absence_importance / 100.0
    count = min(int(absences), 4)

    role_attack_penalty = {
        "Attackers": 0.030,
        "Midfielders": 0.018,
        "Defenders": 0.006,
        "Goalkeeper": 0.004,
        "Mixed": 0.018,
        "Not specified": 0.010,
    }.get(absence_type, 0.010)

    role_defence_penalty = {
        "Attackers": 0.003,
        "Midfielders": 0.012,
        "Defenders": 0.028,
        "Goalkeeper": 0.040,
        "Mixed": 0.020,
        "Not specified": 0.010,
    }.get(absence_type, 0.010)

    attack *= 1 - role_attack_penalty * count * importance
    defense *= 1 + role_defence_penalty * count * importance

    # Fatigue: 5 days is treated as neutral.
    rest_effect = clamp(
        (rest_days - 5) * 0.006,
        -0.025,
        0.018,
    )

    attack *= 1 + rest_effect
    defense *= 1 - rest_effect * 0.45

    if second_leg:
        if aggregate_for < aggregate_against:
            attack *= 1.045
            defense *= 1.012
        elif aggregate_for > aggregate_against:
            attack *= 0.985
            defense *= 0.992
        else:
            attack *= 1.015

    return (
        clamp(attack, 0.82, 1.20),
        clamp(defense, 0.82, 1.20),
    )

# ============================================================
# EXPECTED GOALS
# ============================================================

def _profile_rate(profile, key, fallback):
    value = safe_float(profile.get(key), fallback)
    return fallback if value is None else max(0.05, value)

def predict_xg(
    home_profile,
    away_profile,
    home_context,
    away_context,
    venue,
    weather,
    pitch,
    environment_factor_override=None,
    learned_bias=None,
):
    """Empirical, data-driven scoring model. Avoids hard-coded 1.50/1.20 lambdas.

    Venue-specific attack/defence rates are blended with overall rates, then
    shrunk toward a competition-neutral prior only when samples are small.
    """
    h_n=max(0,int(home_profile.get("matches",0)))
    a_n=max(0,int(away_profile.get("matches",0)))

    h_home_gf=_profile_rate(home_profile,"home_gf",home_profile.get("gf",1.35))
    h_home_ga=_profile_rate(home_profile,"home_ga",home_profile.get("ga",1.35))
    h_gf=_profile_rate(home_profile,"gf",1.35)
    h_ga=_profile_rate(home_profile,"ga",1.35)
    a_away_gf=_profile_rate(away_profile,"away_gf",away_profile.get("gf",1.25))
    a_away_ga=_profile_rate(away_profile,"away_ga",away_profile.get("ga",1.35))
    a_gf=_profile_rate(away_profile,"gf",1.25)
    a_ga=_profile_rate(away_profile,"ga",1.35)

    # More data => more trust in venue splits.
    hw=min(1.0,h_n/8.0)
    aw=min(1.0,a_n/8.0)
    h_attack=hw*h_home_gf+(1-hw)*h_gf
    h_defence=hw*h_home_ga+(1-hw)*h_ga
    a_attack=aw*a_away_gf+(1-aw)*a_gf
    a_defence=aw*a_away_ga+(1-aw)*a_ga

    # Goal-rate blend: own scoring and opponent concession.
    # Home advantage is estimated rather than forcing a fixed 1.50 baseline.
    home_raw=0.58*h_attack+0.42*a_defence
    away_raw=0.58*a_attack+0.42*h_defence

    # Mild shrinkage for sparse samples only. This prevents defaults from
    # dominating while keeping tiny samples from producing absurd lambdas.
    h_shrink=max(0.0,1-min(h_n,10)/10)
    a_shrink=max(0.0,1-min(a_n,10)/10)
    home_raw=(1-h_shrink*0.28)*home_raw+(h_shrink*0.28)*1.35
    away_raw=(1-a_shrink*0.28)*away_raw+(a_shrink*0.28)*1.10

    # Recent scoring acceleration: last five vs full weighted sample.
    h_recent=safe_float(home_profile.get("recent_gf5"),h_gf)
    a_recent=safe_float(away_profile.get("recent_gf5"),a_gf)
    if h_gf>0: home_raw*=clamp(1+0.10*((h_recent/h_gf)-1),0.93,1.08)
    if a_gf>0: away_raw*=clamp(1+0.10*((a_recent/a_gf)-1),0.93,1.08)

    home_xg=home_raw*home_context[0]*away_context[1]
    away_xg=away_raw*away_context[0]*home_context[1]

    environment_factor=(WEATHER.get(weather,1.0)*PITCH.get(pitch,1.0))
    if environment_factor_override is not None:
        environment_factor=clamp(float(environment_factor_override),0.86,1.02)
    home_xg*=environment_factor
    away_xg*=environment_factor

    # Online learning from previously completed MATCHLAB predictions.
    if learned_bias:
        home_xg*=learned_bias.get("home_factor",1.0)
        away_xg*=learned_bias.get("away_factor",1.0)

    return clamp(home_xg,0.20,4.80), clamp(away_xg,0.15,4.50)

# ============================================================
# DIXON-COLES / POISSON
# ============================================================

def poisson_pmf(k, lam):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def calculate_matrix(home_xg, away_xg, max_goals=10, rho=-0.05):
    matrix = []
    total = 0.0

    for hg in range(max_goals + 1):
        row = []
        for ag in range(max_goals + 1):
            p = (
                poisson_pmf(hg, home_xg)
                * poisson_pmf(ag, away_xg)
            )

            if hg == 0 and ag == 0:
                tau = 1 - home_xg * away_xg * rho
            elif hg == 0 and ag == 1:
                tau = 1 + home_xg * rho
            elif hg == 1 and ag == 0:
                tau = 1 + away_xg * rho
            elif hg == 1 and ag == 1:
                tau = 1 - rho
            else:
                tau = 1.0

            p *= max(tau, 0.01)
            row.append(p)
            total += p

        matrix.append(row)

    if total <= 0:
        raise RuntimeError("Probability matrix could not be normalized.")

    matrix = [
        [p / total for p in row]
        for row in matrix
    ]

    home_win = 0.0
    draw = 0.0
    away_win = 0.0

    markets = Counter()
    home_clean = 0.0
    away_clean = 0.0

    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            p = matrix[hg][ag]

            if hg > ag:
                home_win += p
            elif hg == ag:
                draw += p
            else:
                away_win += p

            goals = hg + ag

            for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
                if goals > line:
                    markets[f"over_{str(line).replace('.', '')}"] += p

            if hg > 0 and ag > 0:
                markets["btts_yes"] += p

            if ag == 0:
                home_clean += p

            if hg == 0:
                away_clean += p


    return {
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "over_05": markets["over_05"],
        "over_15": markets["over_15"],
        "over_25": markets["over_25"],
        "over_35": markets["over_35"],
        "over_45": markets["over_45"],
        "under_05": 1 - markets["over_05"],
        "under_15": 1 - markets["over_15"],
        "under_25": 1 - markets["over_25"],
        "under_35": 1 - markets["over_35"],
        "under_45": 1 - markets["over_45"],
        "btts_yes": markets["btts_yes"],
        "btts_no": 1 - markets["btts_yes"],
        "home_clean_sheet": home_clean,
        "away_clean_sheet": away_clean,
        "home_or_draw": home_win + draw,
        "draw_or_away": draw + away_win,
        "home_or_away": home_win + away_win,
        "matrix": matrix,
    }


def poisson_sample(lam, rng):
    if lam <= 0:
        return 0

    threshold = math.exp(-lam)
    product = 1.0
    k = 0

    while product > threshold:
        k += 1
        product *= rng.random()

    return k - 1


def run_monte_carlo(home_xg, away_xg, n_sims, seed=42):
    rng = random.Random(seed)
    counts = Counter()
    total_goals = 0

    for _ in range(n_sims):
        hg = poisson_sample(home_xg, rng)
        ag = poisson_sample(away_xg, rng)

        total_goals += hg + ag

        if hg > ag:
            counts["home_win"] += 1
        elif hg == ag:
            counts["draw"] += 1
        else:
            counts["away_win"] += 1

        if hg + ag >= 1:
            counts["over_05"] += 1
        if hg + ag >= 2:
            counts["over_15"] += 1
        if hg + ag >= 3:
            counts["over_25"] += 1
        if hg + ag >= 4:
            counts["over_35"] += 1
        if hg + ag >= 5:
            counts["over_45"] += 1
        if hg > 0 and ag > 0:
            counts["btts_yes"] += 1

    result = {
        key: counts[key] / n_sims
        for key in [
            "home_win",
            "draw",
            "away_win",
            "over_05",
            "over_15",
            "over_25",
            "over_35",
            "over_45",
            "btts_yes",
        ]
    }

    result["samples"] = n_sims
    result["average_goals"] = total_goals / n_sims
    return result

# ============================================================
# EXPLANATION ENGINE
# ============================================================

def build_explanation(
    home_team,
    away_team,
    home_profile,
    away_profile,
    home_xg,
    away_xg,
    matrix_result,
    injuries_home,
    injuries_away,
    h2h,
    home_context,
    away_context,
):
    reasons = []
    risks = []

    home_attack_edge = home_profile["gf"] - away_profile["ga"]
    away_attack_edge = away_profile["gf"] - home_profile["ga"]

    if home_attack_edge > 0.25:
        reasons.append(
            f"{home_team} has the stronger recent scoring-versus-opponent-defence profile: "
            f"{home_profile['gf']:.2f} goals per match versus {away_profile['ga']:.2f} conceded."
        )
    elif away_attack_edge > 0.25:
        reasons.append(
            f"{away_team} has the stronger recent attacking-versus-defence matchup: "
            f"{away_profile['gf']:.2f} scored versus {home_profile['ga']:.2f} conceded."
        )

    if home_profile["points"] - away_profile["points"] > 0.35:
        reasons.append(
            f"Recent weighted points favour {home_team} "
            f"({home_profile['points']:.2f} per match versus {away_profile['points']:.2f})."
        )
    elif away_profile["points"] - home_profile["points"] > 0.35:
        reasons.append(
            f"Recent weighted points favour {away_team} "
            f"({away_profile['points']:.2f} per match versus {home_profile['points']:.2f})."
        )

    if home_profile["home_gf"] > home_profile["away_gf"] + 0.15:
        reasons.append(
            f"{home_team}'s recent home scoring rate is stronger than its overall rate."
        )

    if away_profile["away_gf"] > away_profile["home_gf"] + 0.15:
        reasons.append(
            f"{away_team} has shown a meaningful away-scoring signal in its recent sample."
        )

    if home_xg > away_xg + 0.35:
        reasons.append(
            f"The combined model produces a sizeable expected-goals advantage for {home_team}: "
            f"{home_xg:.2f} xG versus {away_xg:.2f}."
        )
    elif away_xg > home_xg + 0.35:
        reasons.append(
            f"The combined model produces a sizeable expected-goals advantage for {away_team}: "
            f"{away_xg:.2f} xG versus {home_xg:.2f}."
        )

    if injuries_home:
        risks.append(
            f"{home_team} has {len(injuries_home)} reported availability issue(s); "
            "their exact importance depends on player role and minutes."
        )

    if injuries_away:
        risks.append(
            f"{away_team} has {len(injuries_away)} reported availability issue(s); "
            "their exact importance depends on player role and minutes."
        )

    if matrix_result["draw"] > 0.25:
        risks.append(
            "The draw probability is materially high, so the leading team outcome is not dominant."
        )

    if abs(home_xg - away_xg) < 0.25:
        risks.append(
            "Expected goals are close, meaning the match is structurally competitive and sensitive to small events."
        )

    if matrix_result["btts_yes"] > 0.60:
        reasons.append(
            "Both-teams-to-score probability is elevated, suggesting attacking contributions from both sides."
        )

    if matrix_result["over_25"] > 0.58:
        reasons.append(
            "The model's total-goals distribution leans toward three or more goals."
        )

    if not reasons:
        reasons.append(
            "The result is primarily driven by the combined recent-strength and expected-goals estimates rather than one extreme input."
        )

    if not risks:
        risks.append(
            "No single counter-factor dominates the model, but football outcomes remain uncertain even when the probability gap is meaningful."
        )

    return reasons, risks

# ============================================================
# DATA COVERAGE
# ============================================================

def coverage_score(home_profile, away_profile, fixture, injuries_home, injuries_away, h2h):
    score = 30

    matches = home_profile["matches"] + away_profile["matches"]
    score += min(25, matches)

    if fixture:
        score += 10

    if injuries_home is not None:
        score += 7

    if injuries_away is not None:
        score += 7

    if h2h:
        score += 5

    if fixture and fixture.get("fixture", {}).get("venue", {}).get("name"):
        score += 4

    score = int(clamp(score, 0, 100))

    if score >= 80:
        level = "High"
    elif score >= 60:
        level = "Moderate"
    else:
        level = "Limited"

    return score, level


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## ⚽ MATCHLAB")
    st.caption("NAVY / YELLOW · ADAPTIVE FOOTBALL ANALYST")
    page = st.radio("Workspace", ["Analyze Fixture", "Prediction History"], index=0)
    st.divider()
    st.subheader("Decision style")
    decision_style=st.selectbox("Prediction mode", ["Balanced", "Conservative"], index=0)
    show_research=st.toggle("Show detailed research", value=True)
    show_diagnostics=st.toggle("Show technical diagnostics", value=False)
    st.divider()
    st.subheader("MATCHLAB engine")
    history_count=load_history(1000)
    completed_count=int(history_count["result_checked"].sum()) if not history_count.empty else 0
    st.metric("Completed matches learned", completed_count)
    st.success("Automatic result syncing: ON")
    st.success("Live web research: ON")
    st.success("Q&A can research fresh news: ON")
    st.success("No manual bookmaker odds")
    st.success("No correct-score betting")
    st.caption("The internal probability matrix is still used to calculate 1X2, goals and BTTS probabilities. You no longer need to control its goal limit.")
    with st.expander("League & data coverage"):
        st.write("**Major leagues:** Premier League · LaLiga · Serie A · Bundesliga · Ligue 1")
        st.write("**Also:** many domestic, continental and international competitions when free data can verify the fixture.")
        st.caption("MATCHLAB is not restricted to a fixed team list. Enter any real fixture; the engine searches structured data first and uses public research as a fallback.")
    with st.expander("What MATCHLAB researches"):
        st.write("• Recent form and home/away splits")
        st.write("• Opponent context and H2H")
        st.write("• Injuries, suspensions and team news")
        st.write("• Future fixtures, rest and rotation risk")
        st.write("• Stadium weather")
        st.write("• Motivation and competition context")
        st.write("• Historical calibration and learning")


# ============================================================
# GLOBAL FOOTBALL AI
# ============================================================
# This assistant is deliberately independent of the currently selected fixture.
# A user can ask about any club, player, league, match, transfer, table, or football topic.
with st.sidebar:
    st.divider()
    st.subheader("🤖 Ask Football AI")
    st.caption("Ask about ANY football topic — not just the match being analyzed. MATCHLAB will research the public web when needed.")
    if "global_football_chat" not in st.session_state:
        st.session_state["global_football_chat"]=[]
    for msg in st.session_state["global_football_chat"][-3:]:
        st.markdown(f"**You:** {msg['q']}")
        st.markdown(f"**MATCHLAB:** {msg['a']}")
    global_question=st.text_input("Football question", placeholder="e.g. Is Manchester City playing tomorrow?", key="global_football_question")
    global_ask=st.button("Research & Answer", type="primary", use_container_width=True, key="global_football_ask")
    if global_ask and global_question.strip():
        with st.spinner("MATCHLAB is researching football sources…"):
            packet=global_football_research(global_question.strip())
            global_answer=answer_global_football_question(global_question.strip(),packet)
        st.session_state["global_football_chat"].append({"q":global_question.strip(),"a":global_answer,"packet":packet})
        st.rerun()

if page == "Prediction History":
    st.markdown('<div class="title">MATCHLAB History</div>', unsafe_allow_html=True)
    st.caption("Previous predictions saved by this MATCHLAB instance.")
    history_df = load_history(200)
    if history_df.empty:
        st.info("No previous predictions have been saved yet.")
        st.stop()
    display_df = history_df.copy()
    display_df["Prediction"] = display_df["predicted_outcome"]
    display_df["Model"] = display_df[
        ["home_probability", "draw_probability", "away_probability"]
    ].max(axis=1).map(lambda x: f"{x * 100:.1f}%")
    columns = [
        "id","created_at","match_date","competition","home_team","away_team",
        "Prediction","Model","confidence_score","result_checked"
    ]
    st.dataframe(display_df[columns], use_container_width=True, hide_index=True)
    st.subheader("Check a Previous Prediction")
    selected_id = st.selectbox("Prediction ID", display_df["id"].tolist())
    selected = display_df[display_df["id"] == selected_id].iloc[0]
    st.markdown(f"### {selected['home_team']} vs {selected['away_team']}")
    c1,c2,c3=st.columns(3)
    with c1: st.metric("Home", pct(selected["home_probability"]))
    with c2: st.metric("Draw", pct(selected["draw_probability"]))
    with c3: st.metric("Away", pct(selected["away_probability"]))
    if int(selected["result_checked"] or 0) == 1:
        st.success(
            f"Actual result recorded: {int(selected['actual_home_goals'])}-"
            f"{int(selected['actual_away_goals'])}"
        )
    else:
        st.info("Actual result has not been recorded yet.")
    st.caption("Final results are synchronized automatically from the football-data source when available.")
    if st.button("Delete Selected Prediction"):
        delete_prediction(selected_id)
        st.success("Prediction deleted.")
        st.rerun()
    st.stop()

st.markdown("""
<div class="hero">
  <h1>⚽ MATCHLAB <span style="opacity:.65">LIVE</span></h1>
  <p>Independent football probability engine · live form · team news · future fixtures · rotation risk · weather · calibrated value</p>
</div>
""", unsafe_allow_html=True)

st.subheader("1. Real-World Fixture")
f1,f2=st.columns(2)
with f1:
    home_team_input=st.text_input("Home team",placeholder="e.g. Arsenal")
with f2:
    away_team_input=st.text_input("Away team",placeholder="e.g. Chelsea")
f3,f4=st.columns(2)
with f3:
    match_date=st.date_input("Match date",value=date.today()+timedelta(days=1))
with f4:
    competition_hint=st.text_input(
        "Competition / league hint (optional)",
        placeholder="e.g. Premier League / MLS",
    )
# Internal numerical settings are intentionally fixed. The score matrix is an engine component, not a user betting control.
max_goals=10
simulations=25000

analyze=st.button("ANALYZE FIXTURE",type="primary",use_container_width=True)

if analyze:
    if not home_team_input.strip() or not away_team_input.strip():
        st.error("Enter both teams.")
        st.stop()
    if norm(home_team_input)==norm(away_team_input):
        st.error("Home and away teams must be different.")
        st.stop()

    with st.status("MATCHLAB is researching the fixture...", expanded=True) as status:
        st.write("1/5 Verifying the real fixture with free football data and public web research...")
        try:
            fixture,home_team,away_team=find_fixture(
                home_team_input.strip(),away_team_input.strip(),match_date,competition_hint
            )
        except Exception as error:
            status.update(label="Fixture verification failed",state="error")
            st.error(str(error))
            st.info(
                "If the free football database does not cover this fixture, "
                "MATCHLAB can still use Internet research, but it will not invent a fixture."
            )
            st.stop()

        fixture_id=fixture["fixture"]["id"]
        st.write("2/5 Collecting recent form and venue-specific results...")
        home_fixtures=collect_recent_events(home_team["id"],days=180)
        away_fixtures=collect_recent_events(away_team["id"],days=180)
        home_profile=build_recent_team_profile(
            [{"fixture":{"date":x["date"]+"T00:00:00","status":{"short":"FT"}},"teams":{"home":{"id":x["team_id"] if x["venue"]=="home" else x["opponent_id"]},"away":{"id":x["opponent_id"] if x["venue"]=="home" else x["team_id"]}},"goals":{"home":x["gf"] if x["venue"]=="home" else x["ga"],"away":x["ga"] if x["venue"]=="home" else x["gf"]},"league":{"name":x["competition"]}} for x in home_fixtures],
            home_team["id"]
        )
        away_profile=build_recent_team_profile(
            [{"fixture":{"date":x["date"]+"T00:00:00","status":{"short":"FT"}},"teams":{"home":{"id":x["team_id"] if x["venue"]=="home" else x["opponent_id"]},"away":{"id":x["opponent_id"] if x["venue"]=="home" else x["team_id"]}},"goals":{"home":x["gf"] if x["venue"]=="home" else x["ga"],"away":x["ga"] if x["venue"]=="home" else x["gf"]},"league":{"name":x["competition"]}} for x in away_fixtures],
            away_team["id"]
        )

        st.write("3/6 Researching injuries, lineups, team news and future fixtures...")
        web_research=public_web_research(
            home_team["name"],away_team["name"],match_date.isoformat(),competition_hint
        )
        team_news=research_team_news(home_team["name"],away_team["name"],match_date.isoformat(),competition_hint)
        injuries_home=web_injury_items(web_research,home_team["name"])
        injuries_away=web_injury_items(web_research,away_team["name"])
        home_schedule=future_fixture_research(home_team["name"],home_team.get("id"),match_date,competition_hint)
        away_schedule=future_fixture_research(away_team["name"],away_team.get("id"),match_date,competition_hint)

        st.write("4/6 Checking host-stadium weather and match motivation...")
        weather_data=fetch_stadium_weather(
            fixture.get("fixture",{}).get("venue",{}).get("name"),
            home_team["name"],
            fixture.get("fixture",{}).get("date"),
            match_date,
        )
        motivation_data=research_motivation(
            home_team["name"],away_team["name"],competition_hint,web_research
        )
        st.write("5/6 Building the independent prediction...")
        st.write("6/6 Applying congestion, rotation and uncertainty safeguards...")
        status.update(label="Live research complete",state="complete")

    venue_name=fixture.get("fixture",{}).get("venue",{}).get("name") or "Unknown venue"
    competition=fixture.get("league",{}).get("name") or competition_hint or "Unknown competition"
    h2h=collect_h2h_from_web(home_team["name"],away_team["name"])

    motivation_attack_home,motivation_attack_away=motivation_adjustments(motivation_data)
    home_rest_days=calculated_rest_days(home_profile,match_date)
    away_rest_days=calculated_rest_days(away_profile,match_date)
    # Team-news penalties are deliberately conservative until player-level role
    # and minutes data are available. Evidence can move the model, but cannot
    # overpower the underlying form/strength model.
    home_absence_importance=min(65,35+8*len(injuries_home))
    away_absence_importance=min(65,35+8*len(injuries_away))
    home_context=context_multiplier(
        "4-3-3",motivation_data.get("home",{}).get("level","Normal"),len(injuries_home),"Mixed",home_absence_importance,
        home_profile,home_rest_days,0,0,False
    )
    away_context=context_multiplier(
        "4-3-3",motivation_data.get("away",{}).get("level","Normal"),len(injuries_away),"Mixed",away_absence_importance,
        away_profile,away_rest_days,0,0,False
    )
    home_rotation_factor,home_rotation_note=schedule_adjustment(home_schedule,team_news.get("home",{}))
    away_rotation_factor,away_rotation_note=schedule_adjustment(away_schedule,team_news.get("away",{}))
    home_context=(home_context[0]*motivation_attack_home*home_rotation_factor,home_context[1])
    away_context=(away_context[0]*motivation_attack_away*away_rotation_factor,away_context[1])
    environment_factor=weather_data.get("environment_factor") if weather_data.get("available") else 1.0
    # Learn only from completed predictions that pre-date this fixture.
    # This must be computed before xG prediction so the learned correction
    # is actually available to predict_xg().
    learned_bias=learned_goal_bias(match_date)
    home_xg,away_xg=predict_xg(
        home_profile,away_profile,home_context,away_context,
        "Home team venue",
        weather_data.get("label","Clear") if weather_data.get("available") else "Clear",
        "Good",
        environment_factor_override=environment_factor,
        learned_bias=learned_bias,
    )
    matrix=calculate_matrix(home_xg,away_xg,max_goals=max_goals,rho=-0.05)
    matrix,calibration=calibrate_1x2(matrix,match_date)
    simulation=run_monte_carlo(home_xg,away_xg,simulations,seed=42)
    stability=matrix_simulation_stability(matrix,simulation)
    safe_market,safe_probability,safe_options=safest_double_chance(matrix)
    decision=choose_primary_decision(matrix,home_team["name"],away_team["name"],55,model_uncertainty(55,(home_profile,away_profile),matrix,fixture.get("raw_event",{}).get("provisional",False),0),fixture.get("raw_event",{}).get("provisional",False),decision_style)


    st.session_state['matchlab_analysis'] = {
        'fixture': fixture,
        'home_team': home_team,
        'away_team': away_team,
        'fixture_id': fixture_id,
        'home_profile': home_profile,
        'away_profile': away_profile,
        'web_research': web_research,
        'injuries_home': injuries_home,
        'injuries_away': injuries_away,
        'venue_name': fixture.get('fixture', {}).get('venue', {}).get('name') or 'Unknown venue',
        'competition': fixture.get('league', {}).get('name') or competition_hint or 'Unknown competition',
        'h2h': h2h,
        'home_context': home_context,
        'away_context': away_context,
        'home_xg': home_xg,
        'away_xg': away_xg,
        'matrix': matrix,
        'simulation': simulation,
        'weather_data': weather_data,
        'motivation_data': motivation_data,
        'team_news': team_news,
        'home_schedule': home_schedule,
        'away_schedule': away_schedule,
        'home_rotation_note': home_rotation_note,
        'away_rotation_note': away_rotation_note,
        'learned_bias': learned_bias,
        'calibration': calibration,
        'safe_market': safe_market,
        'safe_probability': safe_probability,
        'safe_options': safe_options,
        'decision': decision,
        'stability': stability,
        'home_rest_days': home_rest_days,
        'away_rest_days': away_rest_days,
        'input_signature': (
            norm(home_team_input), norm(away_team_input),
            match_date.isoformat(), competition_hint.strip(),
        ),
    }

analysis = st.session_state.get('matchlab_analysis')
current_signature = (
    norm(home_team_input), norm(away_team_input),
    match_date.isoformat(), competition_hint.strip(),
)

if analysis and analysis.get('input_signature') == current_signature:
    fixture = analysis['fixture']
    home_team = analysis['home_team']
    away_team = analysis['away_team']
    fixture_id = analysis['fixture_id']
    home_profile = analysis['home_profile']
    away_profile = analysis['away_profile']
    web_research = analysis['web_research']
    injuries_home = analysis['injuries_home']
    injuries_away = analysis['injuries_away']
    venue_name = analysis['venue_name']
    competition = analysis['competition']
    h2h = analysis['h2h']
    home_context = analysis['home_context']
    away_context = analysis['away_context']
    home_xg = analysis['home_xg']
    away_xg = analysis['away_xg']
    matrix = analysis['matrix']
    simulation = analysis['simulation']
    weather_data = analysis.get('weather_data', {'available': False, 'reason': 'Not stored'})
    motivation_data = analysis.get('motivation_data', {'home': {'score': 0, 'level': 'Normal/unclear', 'evidence': []}, 'away': {'score': 0, 'level': 'Normal/unclear', 'evidence': []}})
    team_news = analysis.get('team_news', {'home':{},'away':{},'results':[]})
    home_schedule = analysis.get('home_schedule', {'fixtures':[]})
    away_schedule = analysis.get('away_schedule', {'fixtures':[]})
    home_rotation_note = analysis.get('home_rotation_note','')
    away_rotation_note = analysis.get('away_rotation_note','')
    learned_bias = analysis.get('learned_bias', {'samples':0,'home_factor':1.0,'away_factor':1.0,'status':'Learning unavailable'})
    calibration = analysis.get('calibration', {'samples':0,'active':False,'status':'Calibration unavailable'})
    safe_market = analysis.get('safe_market')
    safe_probability = analysis.get('safe_probability')
    safe_options = analysis.get('safe_options', {})
    decision = analysis.get('decision', {})
    stability = analysis.get('stability', {})
    home_rest_days = analysis.get('home_rest_days',5)
    away_rest_days = analysis.get('away_rest_days',5)
if analysis and analysis.get('input_signature') == current_signature:
    verified_date = fixture.get("raw_event", {}).get("verified_date")
    requested_date = fixture.get("raw_event", {}).get("requested_date")
    if verified_date and requested_date and verified_date != requested_date:
        st.warning(
            f"Web research found this fixture on {verified_date} rather than "
            f"the entered date {requested_date}. MATCHLAB will use the verified fixture date."
        )
    if fixture.get("raw_event", {}).get("provisional"):
        st.warning(
            f"Fixture accepted provisionally: {home_team['name']} vs {away_team['name']}. "
            "Free fixture verification was unavailable, so MATCHLAB did not invent a different fixture."
        )
    else:
        st.success(f"Fixture verified: {home_team['name']} vs {away_team['name']}")
    meta1,meta2,meta3=st.columns(3)
    with meta1: st.metric("Competition",competition)
    with meta2: st.metric("Venue",venue_name)
    with meta3: st.metric("Data source","Free + web")

    st.subheader("2. Live Match Conditions")
    lc1,lc2,lc3=st.columns(3)
    with lc1:
        if weather_data.get("available"):
            st.metric("Stadium weather",weather_data.get("label","Variable"))
            st.caption(f"{weather_data.get('temperature_c'):.1f}°C · {weather_data.get('precipitation_mm',0):.1f} mm rain · {weather_data.get('wind_kmh',0):.0f} km/h wind")
        else:
            st.metric("Stadium weather","Unavailable")
            st.caption(weather_data.get("reason","No reliable weather data"))
    with lc2:
        st.metric(f"{home_team['name']} motivation",motivation_data["home"]["level"])
        st.caption("Evidence-based signal; weak evidence stays neutral")
    with lc3:
        st.metric(f"{away_team['name']} motivation",motivation_data["away"]["level"])
        st.caption("Evidence-based signal; weak evidence stays neutral")

    st.subheader("3. MATCHLAB Prediction")
    probs={
        home_team["name"]:matrix["home_win"],
        "Draw":matrix["draw"],
        away_team["name"]:matrix["away_win"],
    }
    predicted_outcome=max(probs,key=probs.get)
    safe_market = safe_market or safest_double_chance(matrix)[0]
    safe_probability = safe_probability if safe_probability is not None else safest_double_chance(matrix)[1]
    # Re-evaluate the actionable decision with the final coverage/uncertainty below; this first view is provisional.

    p1,p2,p3=st.columns(3)
    with p1: st.metric(home_team["name"],pct(matrix["home_win"]))
    with p2: st.metric("Draw",pct(matrix["draw"]))
    with p3: st.metric(away_team["name"],pct(matrix["away_win"]))
    x1,x2,x3=st.columns(3)
    with x1: st.metric(f"{home_team['name']} xG proxy",f"{home_xg:.2f}")
    with x2: st.metric("Total goals proxy",f"{home_xg+away_xg:.2f}")
    with x3: st.metric(f"{away_team['name']} xG proxy",f"{away_xg:.2f}")
    st.markdown(f"### MATCHLAB call: **{decision.get('decision','Analysing…')}**")
    st.markdown(f"**Decision probability:** {pct(decision.get('probability',max(probs.values())))} · **Safest double chance:** {safe_market} ({pct(safe_probability)})")
    st.caption("MATCHLAB now chooses one actionable result or goals market without bookmaker odds. Exact-score betting has been removed.")

    reasons,risks=build_explanation(
        home_team["name"],away_team["name"],home_profile,away_profile,
        home_xg,away_xg,matrix,injuries_home,injuries_away,[],home_context,away_context
    )
    if weather_data.get("available") and weather_data.get("environment_factor",1.0) < 0.985:
        reasons.append(
            f"Host-stadium weather is restrictive ({weather_data.get('label')}, "
            f"{weather_data.get('wind_kmh',0):.0f} km/h wind, {weather_data.get('precipitation_mm',0):.1f} mm precipitation), "
            "so MATCHLAB reduces the total scoring environment rather than pretending conditions are neutral."
        )
    for team_key,team_name in (("home",home_team["name"]),("away",away_team["name"])):
        m=motivation_data.get(team_key,{})
        if m.get("level") in {"High","Low"}:
            reasons.append(f"Public reporting gives {team_name} a {m.get('level').lower()} motivation signal; the effect is deliberately capped so motivation cannot overpower performance data.")
    st.subheader("4. Why MATCHLAB Produced This Result")
    for i,r in enumerate(reasons,1):
        st.markdown(f"<div class='reason'><b>{i}.</b> {r}</div>",unsafe_allow_html=True)
    st.subheader("5. Why Other Outcomes Could Still Happen")
    for i,r in enumerate(risks,1):
        st.markdown(f"<div class='reason'><b>{i}.</b> {r}</div>",unsafe_allow_html=True)

    st.subheader("6. Recent Evidence")
    evidence_df=pd.DataFrame([
        {"Metric":"Recent matches",home_team["name"]:home_profile["matches"],away_team["name"]:away_profile["matches"]},
        {"Metric":"Weighted goals scored",home_team["name"]:round(home_profile["gf"],2),away_team["name"]:round(away_profile["gf"],2)},
        {"Metric":"Weighted goals conceded",home_team["name"]:round(home_profile["ga"],2),away_team["name"]:round(away_profile["ga"],2)},
        {"Metric":"Weighted points per match",home_team["name"]:round(home_profile["points"],2),away_team["name"]:round(away_profile["points"],2)},
        {"Metric":"Home/Away scoring rate",home_team["name"]:round(home_profile["home_gf"],2),away_team["name"]:round(away_profile["away_gf"],2)},
        {"Metric":"Home/Away conceded rate",home_team["name"]:round(home_profile["home_ga"],2),away_team["name"]:round(away_profile["away_ga"],2)},
    ])
    st.dataframe(evidence_df,use_container_width=True,hide_index=True)

    st.subheader("7. Weather & Motivation Evidence")
    if weather_data.get("available"):
        weather_df=pd.DataFrame([{
            "Stadium":venue_name,"Condition":weather_data.get("label"),
            "Temperature °C":weather_data.get("temperature_c"),
            "Rain mm":weather_data.get("precipitation_mm"),
            "Max wind km/h":weather_data.get("wind_kmh"),
            "Scoring environment":f"{weather_data.get('environment_factor',1.0):.3f}",
            "Source":weather_data.get("source",""),
        }])
        st.dataframe(weather_df,use_container_width=True,hide_index=True)
    else:
        st.info("No reliable stadium weather was available, so MATCHLAB did not apply a weather adjustment.")
    mot_rows=[]
    for key,team_name in (("home",home_team["name"]),("away",away_team["name"])):
        m=motivation_data.get(key,{})
        mot_rows.append({"Team":team_name,"Motivation":m.get("level","Normal/unclear"),"Signal score":m.get("score",0),"Evidence items":len(m.get("evidence",[]))})
    st.dataframe(pd.DataFrame(mot_rows),use_container_width=True,hide_index=True)
    mot_evidence=[]
    for key in ("home","away"):
        mot_evidence.extend(motivation_data.get(key,{}).get("evidence",[]))
    if mot_evidence:
        st.dataframe(pd.DataFrame(mot_evidence),use_container_width=True,hide_index=True)
    else:
        st.caption("No strong motivation evidence found; the model remains neutral rather than inventing a narrative.")

    st.subheader("8. Team News & Rotation Risk")
    tn1,tn2=st.columns(2)
    for col,key,team_name in ((tn1,"home",home_team["name"]),(tn2,"away",away_team["name"])):
        with col:
            pack=team_news.get(key,{})
            st.markdown(f"**{team_name}**")
            st.caption(f"Evidence items: {pack.get('evidence_count',0)} · availability reports: {len(pack.get('absence_sources',[]))} · rotation signals: {len(pack.get('rotation_sources',[]))}")
            for fact in pack.get("facts",[])[:5]:
                st.write(f"• {fact.get('Evidence','')[:300]}")
    st.caption("Team-news text is evidence only. MATCHLAB never invents a named injury or confirmed lineup from a search snippet.")

    st.subheader("9. Future Fixtures & Congestion")
    schedule_rows=[]
    for key,team_name,sched,note in (("home",home_team["name"],home_schedule,home_rotation_note),("away",away_team["name"],away_schedule,away_rotation_note)):
        for f in sched.get("fixtures",[])[:6]:
            schedule_rows.append({"Team":team_name,"Date":f.get("date"),"Fixture":f.get("fixture"),"Competition":f.get("competition"),"Days after match":f.get("days_after_target"),"Importance":f.get("importance"),"Source":f.get("source")})
        if note:
            st.info(f"{team_name}: {note}")
    if schedule_rows:
        st.dataframe(pd.DataFrame(schedule_rows),use_container_width=True,hide_index=True)
    else:
        st.warning("No reliable upcoming-fixture schedule was found; no rotation assumption was added.")

    if show_research:
        st.subheader("10. Internet Research Used")
        st.caption(
            "MATCHLAB searched the public Internet automatically. Search evidence is supporting "
            "information; it does not directly overwrite the model probabilities."
        )
        research_rows=[]
        for r in web_research.get("results",[])[:15]:
            research_rows.append({
                "Topic": r.get("query",""),
                "Source": r.get("title",""),
                "Evidence": r.get("content","")[:500],
                "URL": r.get("url",""),
            })
        if research_rows:
            st.dataframe(pd.DataFrame(research_rows),use_container_width=True,hide_index=True)
        else:
            st.warning("Public search returned no usable results. Confidence will be reduced.")

        st.subheader("11. Availability / Team News")
        availability_rows=injuries_home+injuries_away
        if availability_rows:
            st.dataframe(pd.DataFrame(availability_rows),use_container_width=True,hide_index=True)
        else:
            st.info("No clearly reported availability issue was extracted from the public search.")

        st.subheader("12. Previous Head-to-Head Research")
        h2h_rows=[]
        for r in h2h.get("results",[])[:8]:
            h2h_rows.append({
                "Source":r.get("title",""),
                "Evidence":r.get("content","")[:600],
                "URL":r.get("url",""),
            })
        if h2h_rows:
            st.dataframe(pd.DataFrame(h2h_rows),use_container_width=True,hide_index=True)
        else:
            st.info("No usable H2H search results were found.")


    st.subheader("14. Goals & Market Probabilities")
    market_df=pd.DataFrame([
        ["Over 1.5",pct(matrix["over_15"])],["Under 1.5",pct(matrix["under_15"])],
        ["Over 2.5",pct(matrix["over_25"])],["Under 2.5",pct(matrix["under_25"])],
        ["Over 3.5",pct(matrix["over_35"])],["Under 3.5",pct(matrix["under_35"])],
        ["BTTS Yes",pct(matrix["btts_yes"])],["BTTS No",pct(matrix["btts_no"])],
    ],columns=["Market","Probability"])
    st.dataframe(market_df,use_container_width=True,hide_index=True)
    score,level=coverage_score(
        home_profile,away_profile,fixture,injuries_home,injuries_away,
        h2h.get("results",[])
    )
    if len(web_research.get("results",[])) < 5:
        score=int(clamp(score-10,0,100))
    if home_profile["matches"] < 3 or away_profile["matches"] < 3:
        score=int(clamp(score-10,0,100))

    # Conservative live-model gate: a high raw probability is not treated as a betting signal by itself.
    top_prob=max(probs.values())
    sorted_probs=sorted(probs.values(),reverse=True)
    probability_gap=sorted_probs[0]-sorted_probs[1]
    if fixture.get("raw_event",{}).get("provisional"):
        score=int(clamp(score-15,0,100))
    if not weather_data.get("available"):
        score=int(clamp(score-5,0,100))
    if not motivation_data.get("home",{}).get("evidence") and not motivation_data.get("away",{}).get("evidence"):
        score=int(clamp(score-5,0,100))
    if probability_gap < 0.06:
        score=int(clamp(score-8,0,100))
    # Team-news and future-fixture evidence improve context coverage without overpowering the statistical model.
    if team_news.get("home",{}).get("evidence_count",0) + team_news.get("away",{}).get("evidence_count",0) >= 4:
        score=int(clamp(score+4,0,100))
    if home_schedule.get("fixtures") and away_schedule.get("fixtures"):
        score=int(clamp(score+4,0,100))
    level="High" if score>=80 else "Moderate" if score>=60 else "Limited"
    uncertainty=model_uncertainty(score,(home_profile,away_profile),matrix,fixture.get("raw_event",{}).get("provisional",False),team_news.get("home",{}).get("evidence_count",0)+team_news.get("away",{}).get("evidence_count",0))
    decision=choose_primary_decision(matrix,home_team["name"],away_team["name"],score,uncertainty,fixture.get("raw_event",{}).get("provisional",False),decision_style)
    st.subheader("15. Final Decision")
    st.markdown(f"<div class='decision'><span class='tag'>{decision.get('type','RESULT')}</span><b>{decision.get('decision','')}</b><br><span class='muted'>Probability {pct(decision.get('probability',0))} · Coverage {score}/100 · uncertainty {uncertainty*100:.1f}%</span><br>{decision.get('reason','')}</div>",unsafe_allow_html=True)
    st.caption("This is a model decision, not a guarantee. MATCHLAB does not require bookmaker odds and will not default to NO BET simply because no odds were entered.")

    st.subheader("16. Ask MATCHLAB About This Prediction")
    st.markdown("<div class='ask-box'><b>Talk to MATCHLAB about this prediction.</b><br><span class='muted'>Ask naturally: “Why did you choose them?”, “Should I take Over 1.5?”, “What would make the draw happen?”, “What team news matters?”, “Is this prediction strong?”, or simply “Why?”</span></div>",unsafe_allow_html=True)
    if "matchlab_questions" not in st.session_state:
        st.session_state["matchlab_questions"]=[]
    for item in st.session_state["matchlab_questions"][-6:]:
        st.chat_message("user").write(item["q"])
        st.chat_message("assistant").write(item["a"])
        if item.get("sources"):
            with st.expander("Research sources & method", expanded=False):
                rp=item.get("research",{}) or {}
                st.caption(f"{rp.get('agent','MATCHLAB Research Agent')} · {rp.get('source_count',0)} deduplicated sources · searched {rp.get('searched_at','unknown')}")
                if rp.get("intents"):
                    st.write("Research topics: " + ", ".join(rp.get("intents",[])))
                if rp.get("conflicts"):
                    st.warning("Some topics have conflicting public reports. MATCHLAB keeps those items uncertain and gives higher weight to stronger sources.")
                for url in item["sources"]:
                    st.markdown(f"- [{url}]({url})")
    question=st.chat_input("Ask MATCHLAB anything about football…")
    if question:
        analysis_for_q={
            "home_team":home_team,"away_team":away_team,"matrix":matrix,"home_xg":home_xg,"away_xg":away_xg,
            "decision":decision,"team_news":team_news,"weather_data":weather_data,"learned_bias":learned_bias,
            "calibration":calibration,"home_profile":home_profile,"away_profile":away_profile,
            "reasons":reasons,"risks":risks,
            "conversation":st.session_state.get("matchlab_questions",[])[-4:],
        }
        qn=norm(question)
        fixture_words=[norm(home_team["name"]),norm(away_team["name"]),"this match","this game","the prediction","our prediction","the fixture","this fixture"]
        about_fixture=any(w and w in qn for w in fixture_words)
        # Short football questions such as “Is Manchester City playing tomorrow?” should
        # not be forced into the active fixture context. They use the global football agent.
        with st.spinner("MATCHLAB is researching the public football web…"):
            if about_fixture:
                live_q_research=research_agent(home_team["name"],away_team["name"],match_date.isoformat(),competition,question)
                answer=answer_model_question(question,analysis_for_q,live_q_research)
            else:
                live_q_research=global_football_research(question)
                answer=answer_global_football_question(question,live_q_research)
        st.session_state["matchlab_questions"].append({"q":question,"a":answer,"sources":[r.get("url") for r in live_q_research.get("results",[])[:10] if r.get("url")],"research":live_q_research})
        st.rerun()

    st.subheader("17. Model Learning & Calibration")
    metrics=learning_metrics(match_date)
    lm1,lm2,lm3,lm4=st.columns(4)
    with lm1: st.metric("Completed predictions",str(metrics.get("samples",0)))
    with lm2: st.metric("1X2 Brier",f"{metrics['brier']:.3f}" if metrics.get("brier") is not None else "—")
    with lm3: st.metric("1X2 accuracy",pct(metrics["accuracy"]) if metrics.get("accuracy") is not None else "—")
    with lm4: st.metric("xG total MAE",f"{metrics['xg_mae_total']:.2f}" if metrics.get("xg_mae_total") is not None else "—")
    st.caption(calibration.get("status","Calibration status unavailable"))
    st.caption(f"Safest double-chance signal: **{safe_market} ({pct(safe_probability)})** · rest: {home_team['name']} {home_rest_days}d / {away_team['name']} {away_rest_days}d")

    st.subheader("18. Data Coverage")
    c1,c2,c3,c4=st.columns(4)
    with c1: st.metric("Coverage",f"{score}/100")
    with c2: st.metric("Level",level)
    with c3: st.metric("Recent matches",f"{home_profile['matches']} / {away_profile['matches']}")
    with c4: st.metric("Uncertainty",f"{uncertainty*100:.1f}%")

    if show_diagnostics:
        with st.expander("Technical diagnostics", expanded=False):
            st.write(f"Probability engine stability: **{stability.get('status','Unknown')}**")
            if stability.get("max_gap") is not None:
                st.write(f"Largest matrix vs Monte Carlo probability gap: **{stability['max_gap']*100:.2f} percentage points**")
            st.write("The internal score matrix is a probability engine: it enumerates plausible goal counts and aggregates them into 1X2, totals and BTTS probabilities. It is not a correct-score betting feature.")

    st.subheader("19. Goal Learning")
    lb1,lb2,lb3=st.columns(3)
    with lb1: st.metric("Completed predictions used", str(learned_bias.get("samples",0)))
    with lb2: st.metric("Home goal factor", f"{learned_bias.get('home_factor',1.0):.3f}")
    with lb3: st.metric("Away goal factor", f"{learned_bias.get('away_factor',1.0):.3f}")
    st.caption(learned_bias.get("status","Learning status unavailable"))

    model_source="MATCHLAB Adaptive v3 · free structured data + live public research + calibration + model Q&A"
    record={
        "fixture_id":fixture_id,
        "match_date":fixture["fixture"]["date"],
        "competition":competition,
        "home_team":home_team["name"],
        "away_team":away_team["name"],
        "home_probability":matrix["home_win"],
        "draw_probability":matrix["draw"],
        "away_probability":matrix["away_win"],
        "home_xg":home_xg,
        "away_xg":away_xg,
        "predicted_outcome":decision.get("decision",predicted_outcome),
        "predicted_score":"",
        "confidence_score":score,
        "weather":weather_data,
        "motivation":motivation_data,
        "team_news":team_news,
        "future_fixtures":{"home":home_schedule,"away":away_schedule},
        "rotation_notes":{"home":home_rotation_note,"away":away_rotation_note},
        "uncertainty":uncertainty,
        "calibration":calibration,
        "safe_market":safe_market,
        "safe_probability":safe_probability,
        "safe_options":safe_options,
        "rest_days":{"home":home_rest_days,"away":away_rest_days},
        "decision":decision,
        "stability":stability,
        "model_source":model_source,
        "value_summary":"Model-only decision; bookmaker odds are not used.",
        "reasons":reasons,
        "web_sources":[r.get("url") for r in web_research.get("results",[])[:15]],
    }

    # Every completed analysis is stored automatically so MATCHLAB can learn from it later.
    # Duplicate analyses of the same fixture are not inserted repeatedly.
    existing_prediction_id=prediction_exists(record.get("fixture_id"),record.get("match_date"))
    if not existing_prediction_id:
        save_prediction(record)
        st.success("Prediction automatically recorded. When the final score becomes available, MATCHLAB can use it for future learning.")
    else:
        st.caption(f"Prediction history linked to record #{existing_prediction_id}. Completed results will feed the learning layer.")

    st.divider()
    st.markdown(f"### MATCHLAB final assessment: {decision.get('decision','')}")
    st.write(
        f"MATCHLAB's final model call is **{decision.get('decision','')}** at {pct(decision.get('probability',0))}. "
        f"The scoring model estimates {home_xg:.2f}–{away_xg:.2f} goals. "
        "Team news, form, opponent context, rest, future fixtures, weather and motivation are incorporated when reliable evidence exists. "
        "Bookmaker odds and exact-score betting are deliberately excluded from the prediction workflow."
    )
    st.warning(
        "This is a probabilistic model, not a guarantee. Free data coverage varies by competition, "
        "and some injury, lineup and statistical fields may be missing or delayed."
    )
