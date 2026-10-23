# MATCHLAB LIVE — Adaptive Football Prediction Engine

MATCHLAB is a Streamlit football-analysis application designed to produce **independent pre-match probabilities** from free/public football data and live web research.

It is intentionally built around a conservative principle:

> **A prediction is not the same thing as a bet.**

MATCHLAB can identify a most-likely result, a safer double-chance result, or **PASS / NO BET** when the evidence or data quality is insufficient.

---

## What MATCHLAB does

For a user-entered fixture, MATCHLAB combines:

- Recent team results and form
- Home/away scoring and conceding rates
- Opponent-adjusted context where available
- Recent scoring acceleration
- Team news and injury reports from public web sources
- Future fixtures and congestion/rotation signals
- Rest-day calculation
- Motivation/competition context
- Stadium/weather research when available
- Head-to-head information
- Estimated goal rates
- Poisson/Dixon-Coles score matrix
- Monte Carlo simulation
- Conservative 1X2 calibration from completed MATCHLAB predictions
- Online goal-rate learning from completed predictions
- Double-chance safety assessment
- Optional bookmaker-odds/value analysis

No fixed team list is required.

---

## Important modelling principle

MATCHLAB **does not use bookmaker odds to decide the football prediction**.

The workflow is:

1. Research the fixture.
2. Build the independent football model.
3. Produce probabilities.
4. Identify the most likely outcome.
5. Identify the safer result market separately.
6. Only then compare with bookmaker odds, if supplied.
7. Recommend a bet only when the price and model evidence justify it.

This prevents the model from simply copying the market.

---

## Learning system

MATCHLAB stores completed predictions in `matchlab_history.db`.

When final results become available, the app attempts to synchronize unfinished historical predictions automatically using the free football-data source. Results can also be entered manually from **Prediction History**.

Completed predictions are then used for:

### Goal-rate learning

MATCHLAB calculates the historical difference between predicted goals and actual goals and applies a **small, capped correction** to future goal estimates.

Learning is deliberately conservative. A handful of unusual matches must not completely change the model.

### 1X2 calibration

After enough completed predictions are available, MATCHLAB compares its predicted home/draw/away probabilities with the observed results and applies a limited empirical correction.

This specifically protects against a common failure mode:

> turning a reasonable favourite into an unjustifiably high-confidence favourite while suppressing the draw probability.

### Calibration metrics

The learning diagnostics track:

- Brier score
- 1X2 accuracy
- Total xG absolute error
- Total-goals absolute error
- Number of completed predictions used

---

## Lessons built into the model

MATCHLAB development uses actual prediction tests as feedback.

### Alavés vs Málaga lesson

A team-news adjustment can improve a prediction but can also become too aggressive.

Therefore:

- injury/news adjustments are capped;
- the draw probability must remain meaningful when evidence supports it;
- the model reports the safest market separately from the most likely result;
- calibration matters more than whether a single headline prediction wins.

### Bolton vs Middlesbrough lesson

A correct favourite prediction is useful evidence, but one successful match does **not** prove the model is calibrated.

MATCHLAB therefore evaluates many completed predictions instead of learning from isolated wins.

### Gimnasia vs Sarmiento lesson

A match can have a home favourite while still having a substantial draw probability.

MATCHLAB should not force a strong home-win signal merely because:

- the team is at home;
- recent form is slightly better;
- or the head-to-head record favours it.

When the draw remains plausible, the safer market can be **1X** even if the home win is the single most likely outcome.

---

## Current goal model

MATCHLAB's estimated goals are derived from observed goal rates rather than a permanent `1.50 / 1.20` baseline.

The model blends:

- venue-specific scoring;
- venue-specific conceding;
- overall scoring;
- overall conceding;
- recent scoring acceleration;
- context adjustments;
- weather/environment;
- learned historical goal bias.

### Important limitation

The displayed xG is **MATCHLAB's estimated expected-goal proxy**, not proprietary shot-level xG from a paid provider.

TheSportsDB does not provide the full shot-quality dataset required for a true modern event-level xG model. Public web research can add context, but it should not be presented as equivalent to Opta/StatsBomb-style shot xG.

Future versions can improve this when reliable free shot/xG data is available.

---

## Data sources

MATCHLAB is designed to work without paid API keys.

Current public sources include:

- TheSportsDB free football data
- Public search results
- Google News RSS where available
- DuckDuckGo public search
- OpenStreetMap/Nominatim for stadium geocoding
- Open-Meteo for weather

Source availability can vary by competition, country, fixture and time.

If a source is unavailable, MATCHLAB should reduce confidence rather than invent missing information.

---

## Betting philosophy

MATCHLAB is **not a guaranteed-profit system**.

Football contains substantial randomness, and no model can guarantee a winning bet.

The application therefore prioritizes:

- calibrated probabilities;
- independent predictions;
- value rather than prediction certainty alone;
- conservative confidence;
- explicit uncertainty;
- PASS / NO BET when evidence is insufficient.

A 60% probability does not mean a result must happen. It means that, if the probability is properly calibrated, the event should occur approximately 60% of the time over a large sample of comparable predictions.

---

## Running MATCHLAB locally

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start Streamlit

```bash
streamlit run app.py
```

The app opens in the Streamlit browser interface.

---

## Streamlit Cloud deployment

1. Create or open the GitHub repository.
2. Replace the existing `app.py` with the current MATCHLAB `app.py`.
3. Replace `README.md` with this version.
4. Ensure `requirements.txt` is committed.
5. Deploy the repository as a Streamlit app.

No API-Football, Tavily or Serper key is required by the current architecture.

---

## Main workspace

### Analyze Fixture

Enter:

- Home team
- Away team
- Match date
- Optional competition/league hint
- Maximum score-matrix goals
- Monte Carlo simulation count

Then select **ANALYZE FIXTURE**.

### Prediction History

Shows saved predictions and allows the user to:

- inspect previous probabilities;
- see predicted scorelines;
- record an actual result manually;
- delete a prediction.

Historical results are used by the learning system.

---

## Recommended interpretation of MATCHLAB output

Use the results in this order:

1. **Data coverage** — Is there enough reliable information?
2. **Independent probabilities** — What does the model believe?
3. **Safest result market** — Which double chance protects against the most plausible alternative?
4. **Team news** — Is there a meaningful late change?
5. **Uncertainty** — Is the model too close to call?
6. **Odds/value** — Is the bookmaker price actually attractive?
7. **PASS** — If the evidence or price is inadequate, do nothing.

The safest market is not necessarily the market with the biggest expected return.

---

## Known limitations

MATCHLAB is still an evolving model.

Current limitations include:

- limited free-data coverage for some competitions;
- incomplete player-level minutes/role information;
- no universal confirmed-lineup feed;
- estimated rather than provider-grade shot xG;
- limited team-specific learning compared with a full professional forecasting stack;
- historical calibration improves only as completed MATCHLAB predictions accumulate.

The model should therefore be treated as a **decision-support and research tool**, not a guarantee of betting profit.

---

## Roadmap

The next major improvements should focus on:

1. Dynamic team-strength/Elo ratings
2. Opponent-adjusted attack and defence ratings
3. League-specific priors
4. Player-level availability and expected minutes
5. Confirmed-lineup impact modelling
6. Better shot/xG data when reliable free sources exist
7. Negative-binomial/overdispersion score modelling
8. Market-specific calibration
9. Reliability diagrams and calibration curves
10. Full backtesting by league and market
11. Automatic post-match model reports
12. Stronger PASS / NO BET gating

The goal is not to make MATCHLAB sound more confident. The goal is to make its probabilities **more accurate and better calibrated**.

---

## Version philosophy

MATCHLAB versions should only be considered improvements when testing demonstrates that the change improves out-of-sample performance or calibration.

Adding more features is not automatically an improvement.

**Evidence beats complexity.**
