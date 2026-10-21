# MatchLab 2.0

A mobile-friendly Streamlit prototype for football match probabilities.

## Modes
- **Historical CSV:** estimates expected goals from your uploaded completed-match results.
- **Manual expected goals:** lets you test the Poisson probability engine with chosen expected-goals estimates.

## Required CSV columns
`home_team,away_team,home_goals,away_goals`
Optional recommended column: `date` (YYYY-MM-DD).

Example:
```csv
date,home_team,away_team,home_goals,away_goals
2026-09-20,Arsenal,Newcastle,2,1
```
Use real results from the same competition/season where possible. Do not mix leagues without a clear reason. CSV mode requires at least 10 rows and the selected home team to appear as the home team and selected away team to appear as the away team.

## Run locally
```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy from an Android phone
You can use GitHub in your phone browser to create a repository and upload `app.py`, `requirements.txt`, and this README, then deploy it through Streamlit Community Cloud. See the official quickstart: https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/quickstart

## Important limits
No live API is connected in this version. Historical CSV estimates are a simple baseline based on recent venue results and league averages; they are not an advanced AI model and are not yet backtested. For stronger predictions, the next version should add a licensed data provider, time-ordered historical training, lineup/injury features, a calibrated model, prediction history storage, and proper evaluation.
