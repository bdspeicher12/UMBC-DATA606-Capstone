# App

Interactive web application for the project.

## `app.py` — Streamlit Success Predictor

A **Streamlit** app that trains the model on the TMDB dataset and lets a user enter a hypothetical film's **pre-release** attributes — budget, genre, runtime, release month, language, and number of production companies — and returns a **predicted probability of box-office success** (revenue ≥ 2× budget).

## Run it

From the repository root:

```bash
pip install -r app/requirements.txt
streamlit run app/app.py
```

The app loads `data/tmdb_5000_movies.csv`, applies the same cleaning and feature engineering as the notebooks, trains the best model (Gradient Boosting), and serves predictions. Only pre-release features are used — no data leakage.
