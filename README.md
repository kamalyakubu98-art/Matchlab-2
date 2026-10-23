# MATCHLAB LIVE — Adaptive Football Prediction Engine

MATCHLAB is a Streamlit football-analysis application built to produce **independent, evidence-based pre-match football decisions** using free/public data and live web research.

The current philosophy is simple:

> **Research first. Probability second. Decision third. Learn from the result.**

MATCHLAB does **not** require bookmaker odds, does not ask the user to enter betting figures, and no longer uses exact-score prediction as a headline output.

---

## What MATCHLAB now does

For any real fixture entered by the user, MATCHLAB researches and combines:

- Recent team form
- Home/away scoring and conceding rates
- Opponent context
- Recent scoring acceleration
- Team news
- Injuries and suspensions
- Available lineup information
- Future fixtures
- Rest days and fixture congestion
- Rotation risk when supported by evidence
- Motivation/competition context
- Stadium weather when available
- Head-to-head information
- Estimated expected-goal proxies
- Poisson/Dixon-Coles probability modelling
- Monte Carlo stability checking
- Conservative 1X2 calibration
- Historical goal-rate learning
- Automatic post-match result synchronization

No fixed team list is required.

---

# New decision system

MATCHLAB no longer depends on bookmaker odds to decide what it thinks will happen.

After the research and probability calculation, MATCHLAB chooses **one actionable football call**:

### Result call
- **Home team to win**
- **Away team to win**
- **Match to end in a draw**

### Or goals call
When the 1X2 probabilities are too close, MATCHLAB can select a stronger goals market such as:

- Over 1.5 goals
- Under 2.5 goals
- Over 2.5 goals
- Under 3.5 goals
- BTTS
- BTTS No

The model therefore does not automatically say **NO BET** simply because bookmaker odds were not supplied.

A PASS/NO-BET style gate is no longer the central decision engine. The application always provides its strongest model call, while clearly showing probability, uncertainty and data coverage.

---

# No manual bookmaker odds

Bookmaker odds have been removed from the MATCHLAB prediction workflow.

There are no manual odds fields and no manual stake/value calculation.

This makes the architecture cleaner:

1. Research the match.
2. Predict independently.
3. Choose the strongest result or goals market.
4. Explain the decision.
5. Allow the user to question the model.
6. Record the prediction automatically.
7. Learn from the final result later.

If bookmaker odds are used outside the application, they should be treated as a separate decision layer rather than something that changes MATCHLAB's football prediction.

---

# Exact-score prediction removed

MATCHLAB previously displayed a top correct-score prediction.

That feature has been removed from the user-facing model because exact football scorelines were repeatedly clustering around outcomes such as **1-1**, creating false precision and little practical decision value.

The internal probability engine still considers many possible goal counts, but it no longer presents one exact score as a recommendation.

---

# What is the maximum score matrix?

The **score matrix still works**, but it is now an internal calculation rather than a user-controlled setting.

For example, if MATCHLAB estimates:

- Home = 1.45 expected goals
- Away = 0.95 expected goals

The matrix calculates probabilities across possible goal counts and aggregates them into:

- Home win probability
- Draw probability
- Away win probability
- Over/Under probabilities
- BTTS probabilities
- Clean-sheet probabilities

The matrix is therefore useful even though MATCHLAB no longer displays exact-score recommendations.

### Why the user no longer controls the maximum goals

The previous slider asked users to choose something like 5–12 maximum goals. That is an implementation detail, not meaningful football information.

MATCHLAB now uses an internal fixed calculation range and hides that technical control from the user.

This makes the interface simpler and prevents users from accidentally changing a numerical setting that should be controlled by the model.

---

# Monte Carlo simulation

MATCHLAB also runs an internal Monte Carlo simulation from the estimated goal rates.

It is **not** used to generate a correct-score recommendation.

Instead, MATCHLAB compares the simulation with the deterministic probability matrix to check whether the main probability estimates are stable.

The technical diagnostics can show the difference between the two calculations.

A large disagreement is treated as a warning about model dispersion rather than being hidden.

---

# Ask MATCHLAB about its prediction

Every completed analysis has an **interactive Ask MATCHLAB** conversation. It is designed to behave like a fixture analyst, not a panel that simply prints figures.

You can ask natural follow-up questions such as:

- **Why did you choose Arsenal?**
- **Why not the draw?**
- **What team news mattered most?**
- **Should I take Over 1.5?**
- **What would have to happen for the draw?**
- **What could make your prediction wrong?**
- **Which team has the better attack?**
- **Is this a strong prediction or a close one?**
- **Why are you choosing a goals market instead of a winner?**
- **How did the model learn this?**
- **Why?**

The conversation is grounded in the actual fixture analysis: form, venue splits, team news, injuries, weather, goal estimates, probabilities, calibration, risks and data coverage. The response is therefore specific to the match rather than a generic football explanation.

## Live web research inside Q&A

Ask MATCHLAB is now allowed to perform a **fresh public-web search for the question itself**. It does not only reuse the original pre-match snapshot.

For example, after a prediction you can ask:

- **What is the latest injury news?**
- **Has the lineup been confirmed?**
- **What are reporters saying right now?**
- **Why are people expecting a draw?**
- **Is there any late team news that changes the prediction?**
- **What happened in the latest team training/news conference?**

The Q&A research layer searches multiple public routes, currently **DuckDuckGo HTML + Google News RSS**, and shows the web sources used for the answer. The search is performed at question time and cached briefly to avoid repeatedly hammering public endpoints.

### Research philosophy

MATCHLAB should be **research-first rather than source-limited**. When a question requires information that was not in the original analysis, the Q&A layer is allowed to go back to the public web and investigate it.

However, “no limitations” cannot literally mean unlimited requests: public websites impose rate limits, some pages block automated access, and free data sources can be incomplete. MATCHLAB therefore searches broadly while respecting public-service limits and never fabricates information when a source cannot be verified.

Conflicting reports are treated as uncertainty. Official club/competition sources receive greater weight when identifiable, followed by reputable reporting. Search snippets are evidence to investigate, not proof by themselves.

### Important implementation note

The current app deliberately does **not** require a paid LLM API key. The Ask MATCHLAB analyst is a structured, context-grounded reasoning layer with live public-web research. It can explain the model, investigate fresh information and cite the sources it found without inventing facts.

A future optional LLM connector could make the language more open-ended, but it should **never be allowed to override the underlying football model or turn an unverified web claim into a prediction fact**.

---

# Sidebar controls

The new sidebar contains useful controls instead of numerical betting inputs.

### Prediction mode

**Balanced**

Uses the normal decision thresholds.

**Conservative**

Requires stronger separation before selecting a 1X2 result. If the result probabilities remain too close, MATCHLAB is more likely to choose a strong goals market instead.

### Detailed research

Controls whether the full Internet research, availability and H2H evidence tables are displayed.

### Technical diagnostics

Shows internal model-stability information, including the comparison between the probability matrix and Monte Carlo simulation.

The sidebar also shows the number of completed matches currently available to the learning system.

---

# Learning system

MATCHLAB automatically saves completed analyses to `matchlab_history.db`.

When a previously predicted match has finished, MATCHLAB attempts to retrieve the final score automatically using the free football-data source.

Manual final-score entry is no longer required in the normal workflow.

Completed predictions are then used for learning.

## Goal-rate learning

MATCHLAB compares predicted goals with actual goals and applies a small, capped correction to future goal estimates.

The correction is intentionally restrained so that a small number of unusual matches cannot distort the entire model.

## 1X2 calibration

After enough completed predictions are available, MATCHLAB compares:

- predicted home probability;
- predicted draw probability;
- predicted away probability;

with the actual outcomes.

A limited empirical correction is blended into future predictions.

This specifically addresses an important MATCHLAB lesson:

> **Do not turn a reasonable favourite into an unjustifiably strong favourite while crushing the draw probability.**

## Calibration diagnostics

MATCHLAB tracks:

- 1X2 Brier score
- 1X2 accuracy
- xG absolute error
- total-goals absolute error
- number of completed predictions

The objective is not simply to maximize the number of winning headlines. The objective is to make the probabilities better calibrated over time.

---

# Lessons built into MATCHLAB

MATCHLAB development is based on actual prediction tests rather than adding features for appearance alone.

## Alavés vs Málaga

The Alavés analysis showed that matchday news can legitimately move a prediction, but it should not be allowed to overwhelm the underlying model.

Lessons:

- Keep the draw probability meaningful when evidence supports it.
- Cap injury/news adjustments.
- Separate the strongest 1X2 outcome from the safest market.
- Evaluate probability calibration, not just win/loss headlines.

## Bolton vs Middlesbrough

The Middlesbrough call was successful, but one correct prediction cannot prove model quality.

Lesson:

- Keep accumulating results and evaluate performance over many matches.

## Gimnasia La Plata vs Sarmiento

This match demonstrated why a home favourite can still have a meaningful draw probability.

Lessons:

- Do not force a strong favourite because of home advantage alone.
- Use goals markets when 1X2 probabilities are too close.
- Explain why alternatives remain possible.

---

# Current goal model

MATCHLAB does not use a permanent `1.50 / 1.20` goal baseline.

Estimated goals are derived from observed data and context, including:

- venue-specific scoring;
- venue-specific conceding;
- overall scoring;
- overall conceding;
- recent scoring acceleration;
- context adjustments;
- weather/environment;
- learned historical goal bias.

### Important limitation

The displayed xG is **MATCHLAB's estimated expected-goal proxy**, not provider-grade shot-level xG.

TheSportsDB does not provide the full shot-quality dataset required for a professional event-level xG model. Public research can add context, but it should not be described as equivalent to Opta/StatsBomb-style shot xG.

---

# League and fixture coverage

MATCHLAB is **not restricted to a fixed team list**. The user enters a real fixture and the engine searches for that fixture dynamically.

The architecture is intended to cover major competitions including:

- Premier League
- LaLiga
- Serie A
- Bundesliga
- Ligue 1
- Eredivisie
- Primeira Liga
- MLS
- Major South American leagues
- Domestic cups and other competitions when the free data sources can verify them
- Selected continental and international competitions

For the major leagues, coverage should normally be substantially better than for obscure lower divisions. **Full worldwide coverage cannot be guaranteed** because MATCHLAB relies on free/public data rather than a paid global football feed.

The important rule is: if a fixture cannot be verified or the evidence is too incomplete, MATCHLAB should say that coverage is limited rather than manufacture a result, team or statistic.

## Data sources

MATCHLAB is designed to work without paid API keys.

Current public sources include:

- TheSportsDB free football data
- Public web search
- Google News RSS where available
- DuckDuckGo public search
- OpenStreetMap/Nominatim for stadium geocoding
- Open-Meteo for weather

Coverage varies by competition, country and fixture.

When reliable information is missing, MATCHLAB should reduce confidence rather than invent facts.

---

# User workflow

## Analyze Fixture

Enter only:

- Home team
- Away team
- Match date
- Optional competition/league hint

Then press **ANALYZE FIXTURE**.

There are no bookmaker-odds fields and no correct-score controls.

## Prediction History

The history page shows previous MATCHLAB decisions and their probabilities.

Final results are synchronized automatically when the free football-data source provides them.

---

# Running MATCHLAB locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

---

# Streamlit Cloud deployment

1. Replace the repository's `app.py` with the current MATCHLAB file.
2. Replace `README.md` with this version.
3. Keep `requirements.txt` committed.
4. Deploy through Streamlit Cloud.

No API-Football, Tavily or Serper key is required by the current architecture.

---

# Model philosophy

MATCHLAB is not a guaranteed-profit machine.

Football contains randomness, incomplete information and unusual events.

The goal is therefore not to sound certain. The goal is to produce:

- better probabilities;
- better evidence handling;
- clearer uncertainty;
- useful result/goals decisions;
- transparent explanations;
- automatic learning from completed matches.

A 60% probability means the event should occur roughly 60% of the time over a large sample **if the model is calibrated correctly**. It does not mean the next match must win.

---

# Known limitations

MATCHLAB remains an evolving forecasting system.

Current limitations include:

- incomplete free-data coverage for some competitions;
- limited player-level minutes and role data;
- incomplete confirmed-lineup coverage;
- estimated rather than provider-grade shot xG;
- limited team-specific learning compared with a professional forecasting stack;
- calibration quality depends on the number of completed MATCHLAB predictions;
- public web reports can conflict or arrive late.

These limitations should be reflected in the model's uncertainty and coverage rather than hidden.

---

# Roadmap

The strongest future improvements are:

1. Dynamic team-strength/Elo ratings
2. Opponent-adjusted attack and defence ratings
3. League-specific priors
4. Player-level expected-minutes modelling
5. Confirmed-lineup impact modelling
6. Better free shot/xG data
7. Negative-binomial/overdispersion goal modelling
8. Market-specific calibration
9. Reliability curves
10. Full historical backtesting by league
11. Automatic post-match model reports
12. Better player-role and tactical matchup modelling
13. More advanced model Q&A using structured evidence

The rule remains:

> **Evidence beats complexity.**

A feature should stay only if testing shows that it improves out-of-sample prediction or calibration.
