"""
Movie Box-Office Success Predictor — Streamlit app
DATA 606 Capstone · Bennett Speicher

Enter a hypothetical film's PRE-RELEASE attributes and get a predicted probability
that it will be a financial success (worldwide box-office revenue >= 2x production budget).

Run from the repository root with:
    streamlit run app/app.py
"""
import os
import json
import re
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# --- Locate the dataset relative to this file (robust to where streamlit is launched) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "tmdb_5000_movies.csv")

NUMERIC = ["budget_log", "runtime", "n_genres", "n_production_companies",
           "n_spoken_languages", "year", "budget_per_min_log"]
FLAGS = ["is_english", "is_summer_action_adv", "is_october_horror", "is_holiday_family",
         "is_sequel", "is_major_studio"]
CATEGORICAL = ["primary_genre", "season", "release_window"]
FEATURES = NUMERIC + FLAGS + CATEGORICAL

SEQUEL_KEYWORDS = {"sequel", "based on novel", "based on comic", "based on comic book", "superhero",
                   "marvel comic", "dc comics", "based on young adult novel", "remake", "spin off",
                   "saga", "trilogy"}
MAJORS = {"Universal Pictures", "Warner Bros.", "Paramount Pictures",
          "Twentieth Century Fox Film Corporation", "Columbia Pictures", "Walt Disney Pictures",
          "New Line Cinema", "Metro-Goldwyn-Mayer (MGM)", "DreamWorks SKG", "Lionsgate",
          "Miramax Films", "Touchstone Pictures", "Sony Pictures"}


def _parse_names(s):
    try:
        return [d["name"] for d in json.loads(s)]
    except Exception:
        return []


def _season(m):
    if m in (12, 1, 2):
        return "Winter"
    if m in (3, 4, 5):
        return "Spring"
    if m in (6, 7, 8):
        return "Summer"
    return "Fall"


def _window(m):
    if m in (5, 6, 7, 8):
        return "Summer"
    if m in (11, 12):
        return "Holiday"
    if m in (1, 2):
        return "Dump"
    if m in (9, 10):
        return "Fall"
    return "Spring"


@st.cache_data
def load_clean_data():
    df = pd.read_csv(DATA_PATH)
    d = df[df["status"] == "Released"].copy()
    d = d[(d["budget"] >= 1000) & (d["revenue"] > 0)]   # drop unknown (0) & impossible (<$1,000) budgets
    d = d.dropna(subset=["release_date"]).drop_duplicates(subset=["title", "release_date"]).copy()

    d["genre_list"] = d["genres"].apply(_parse_names)
    d["keyword_list"] = d["keywords"].apply(_parse_names)
    d["company_list"] = d["production_companies"].apply(_parse_names)
    d["primary_genre"] = d["genre_list"].apply(lambda g: g[0] if g else "Unknown")
    d["n_genres"] = d["genre_list"].apply(len)
    d["n_production_companies"] = d["company_list"].apply(len)
    d["n_spoken_languages"] = d["spoken_languages"].apply(lambda s: len(_parse_names(s)))
    d["is_english"] = (d["original_language"] == "en").astype(int)

    d["release_date"] = pd.to_datetime(d["release_date"], errors="coerce")
    d["year"] = d["release_date"].dt.year
    d["month"] = d["release_date"].dt.month
    d["budget_log"] = np.log10(d["budget"])
    d["season"] = d["month"].apply(_season)
    d["release_window"] = d["month"].apply(_window)

    def has(gl, *t):
        return int(any(x in gl for x in t))

    d["is_summer_action_adv"] = d.apply(
        lambda r: int(has(r["genre_list"], "Action", "Adventure") and r["month"] in (5, 6, 7)), axis=1)
    d["is_october_horror"] = d.apply(
        lambda r: int(has(r["genre_list"], "Horror") and r["month"] == 10), axis=1)
    d["is_holiday_family"] = d.apply(
        lambda r: int(has(r["genre_list"], "Family", "Animation") and r["month"] in (11, 12)), axis=1)

    kw_flag = d["keyword_list"].apply(lambda ks: int(any(k in SEQUEL_KEYWORDS for k in ks)))
    title_flag = d["title"].fillna("").apply(
        lambda t: int(bool(re.search(r"\b(2|3|4|II|III|IV|V)\b|Part |Chapter ", t))))
    d["is_sequel"] = ((kw_flag == 1) | (title_flag == 1)).astype(int)
    d["is_major_studio"] = d["company_list"].apply(lambda cs: int(any(c in MAJORS for c in cs)))
    d["budget_per_min_log"] = np.log10(np.maximum(d["budget"] / np.maximum(d["runtime"], 1), 1))

    d["success"] = (d["revenue"] >= 2 * d["budget"]).astype(int)
    # Drop list-valued helper columns so Streamlit's cache can hash/serialize the result
    d = d.drop(columns=["genre_list", "keyword_list", "company_list"])
    return d


@st.cache_resource
def train_model(d):
    """Train the most robust model (tuned Random Forest — best cross-validated ROC-AUC)."""
    pre = ColumnTransformer(transformers=[
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUMERIC),
        ("flag", "passthrough", FLAGS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    model = RandomForestClassifier(n_estimators=800, max_depth=12, min_samples_leaf=5, random_state=42)
    pipe = Pipeline([("prep", pre), ("model", model)])
    pipe.fit(d[FEATURES], d["success"])
    return pipe


def build_feature_row(budget, genre, runtime, month, year, is_english, n_prod,
                      n_genres, n_langs, is_sequel, is_major):
    """Turn user inputs into a one-row DataFrame matching the training features."""
    row = {
        "budget_log": np.log10(max(budget, 1)),
        "runtime": runtime,
        "n_genres": n_genres,
        "n_production_companies": n_prod,
        "n_spoken_languages": n_langs,
        "year": year,
        "budget_per_min_log": np.log10(max(budget / max(runtime, 1), 1)),
        "is_english": int(is_english),
        "is_summer_action_adv": int(genre in ("Action", "Adventure") and month in (5, 6, 7)),
        "is_october_horror": int(genre == "Horror" and month == 10),
        "is_holiday_family": int(genre in ("Family", "Animation") and month in (11, 12)),
        "is_sequel": int(is_sequel),
        "is_major_studio": int(is_major),
        "primary_genre": genre,
        "season": _season(month),
        "release_window": _window(month),
    }
    return pd.DataFrame([row])[FEATURES]


# ----------------------------- UI -----------------------------
def main():
    st.set_page_config(page_title="Movie Box-Office Success Predictor", page_icon="🎬")
    st.title("🎬 Movie Box-Office Success Predictor")
    st.caption("Predicts the probability a film earns at least **2× its budget** — using only pre-release info.")

    data = load_clean_data()
    model = train_model(data)
    genres = sorted(g for g in data["primary_genre"].unique() if g and g != "Unknown")
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]

    st.subheader("Enter the film's details")
    col1, col2 = st.columns(2)
    with col1:
        budget = st.number_input("Production budget (USD)", min_value=100_000, max_value=400_000_000,
                                 value=50_000_000, step=1_000_000)
        genre = st.selectbox("Primary genre", genres,
                             index=genres.index("Action") if "Action" in genres else 0)
        runtime = st.slider("Runtime (minutes)", 60, 240, 110)
        year = st.number_input("Release year", min_value=1990, max_value=2030, value=2016)
    with col2:
        month_name = st.selectbox("Release month", months, index=5)
        is_english = st.checkbox("Primarily English-language", value=True)
        is_sequel = st.checkbox("Sequel / franchise / adaptation", value=False)
        is_major = st.checkbox("Major studio production", value=True)
        n_prod = st.slider("Number of production companies", 1, 10, 2)

    with st.expander("Advanced (optional)"):
        n_genres = st.slider("Number of genres listed", 1, 6, 3)
        n_langs = st.slider("Number of spoken languages", 1, 6, 1)

    if st.button("Predict success", type="primary"):
        month = months.index(month_name) + 1
        X = build_feature_row(budget, genre, runtime, month, year, is_english,
                              n_prod, n_genres, n_langs, is_sequel, is_major)
        proba = float(model.predict_proba(X)[0, 1])
        st.markdown("---")
        st.metric("Predicted probability of success", f"{proba*100:.0f}%")
        st.progress(proba)
        if proba >= 0.5:
            st.success(f"✅ Predicted **Successful** — likely to earn ≥ 2× its budget "
                       f"({_season(month)} / {_window(month)} release).")
        else:
            st.warning(f"⚠️ Predicted **Not Successful** — may not clear the 2× break-even bar "
                       f"({_season(month)} / {_window(month)} release).")
        st.caption("Note: revenue here is worldwide box-office gross and budget excludes marketing, "
                   "so this is a grounded proxy for profitability, not exact accounting profit.")


if __name__ == "__main__":
    main()
