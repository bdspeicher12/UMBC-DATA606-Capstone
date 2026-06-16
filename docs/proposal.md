# Predicting Movie Box-Office Success Before Release

## 1. Title and Author

- **Project Title:** Predicting Movie Box-Office Success Before Release
- Prepared for UMBC Data Science Master Degree Capstone by Dr. Chaojie (Jay) Wang
- **Author:** Bennett Speicher
- **GitHub Repository:** https://github.com/bdspeicher12/UMBC-DATA606-Capstone
- **LinkedIn:** https://www.linkedin.com/in/bennettspeicher/
- **PowerPoint Presentation:** *to be added*
- **YouTube Video:** *to be added*

---

## 2. Background

Producing and releasing a film is an enormous financial bet. A studio commits tens or hundreds of millions of dollars to a production years before a single ticket is sold, and most of that decision is made on incomplete information. While many factors that determine a film's fate (word of mouth, reviews, competing releases) only emerge *after* launch, a number of meaningful signals are known **before release** — the production budget, genre, runtime, planned release timing, language, and the production companies behind it.

This project builds a machine-learning model that predicts whether a movie will be a **financial success** using only information available *before* the film is released. The goal is a decision-support tool: given a film's pre-release profile, estimate the probability that it will earn back enough at the box office to be profitable.

### Why it matters

- **Investment risk.** Greenlighting a film is a high-stakes capital allocation decision. A model that flags the financial odds of a project—based on attributes a studio controls or knows early—supports smarter budgeting and portfolio decisions.
- **Resource planning.** Understanding which attributes (budget level, genre, release window) move the odds of success helps studios and investors set realistic expectations and structure marketing spend.
- **A clear, honest framing.** Rather than predicting raw revenue (which is dominated by a few blockbusters and easy to over-fit), this project predicts a **budget-relative** outcome that reflects what studios actually care about: did the movie make money?

### Defining "success"

A movie is labeled a **success** if its worldwide box-office **revenue is at least twice its production budget** (`revenue ≥ 2 × budget`). This 2× rule is a widely used industry proxy for theatrical break-even, for two reasons:

1. **Marketing roughly doubles the cost.** Studios typically spend an additional ~50–100% of the production budget on prints and advertising (P&A).
2. **Theaters keep about half the gross.** Exhibitors retain roughly 50% of box-office ticket sales, so the studio nets only about half of the reported revenue.

Together these mean a film must gross roughly twice its production budget to break even. *Caveat:* the dataset's `revenue` is worldwide box-office gross and `budget` excludes marketing, so this label is a well-grounded **proxy** for profitability rather than exact accounting profit — a limitation acknowledged throughout the project.

### Research questions

1. Can a film's **financial success** (revenue ≥ 2× budget) be predicted from **pre-release** attributes alone?
2. Which pre-release factors—**budget, genre, runtime, release month, language, number of production companies**—most influence the odds of success?
3. Does **release timing** (e.g., summer/holiday windows) meaningfully change the probability of success?
4. How accurately can common classification models distinguish successes from non-successes, and which performs best?

---

## 3. Data

### Data source

- **TMDB 5000 Movie Dataset** — movie metadata from The Movie Database (TMDb), a widely used public film dataset originally released for the Kaggle "TMDB Box Office Prediction" project.
- File used: `data/tmdb_5000_movies.csv`
- Source/origin: The Movie Database (TMDb) via Kaggle.

### Data size and shape

| Property | Value |
| --- | --- |
| File size | ~5.7 MB |
| Raw rows (movies) | 4,803 |
| Raw columns | 20 |
| Usable rows after cleaning | 3,228 (≈67%) |
| Time period | Release years **1916–2016** |

**What each row represents:** one **movie**.

**Cleaning applied:** kept only films with `status = "Released"` and with a **real budget and revenue** (both `> 0`); in TMDB a `0` typically encodes "unknown" and would corrupt the budget-relative target. JSON-encoded columns (e.g., `genres`) were parsed into usable values. After cleaning, **3,228 movies** remain, with a **balanced target** (≈56% successful).

### Data dictionary

| Column | Type | Definition | Example / values |
| --- | --- | --- | --- |
| `budget` | int | Production budget (USD) | 237,000,000 |
| `genres` | JSON | List of genre tags | Action, Adventure, Fantasy |
| `homepage` | string | Official film URL (sparse, ~64% missing) | http://… |
| `id` | int | TMDB movie ID | 19995 |
| `keywords` | JSON | Plot keyword tags | "culture clash", "future" |
| `original_language` | string | ISO 639-1 language code | en, fr, ja |
| `original_title` | string | Title in original language | Avatar |
| `overview` | string | Short plot synopsis | free text |
| `popularity` | float | TMDB popularity score *(post-release)* | 150.4 |
| `production_companies` | JSON | Companies that produced the film | Ingenious Film Partners |
| `production_countries` | JSON | Countries of production | United States of America |
| `release_date` | date | Theatrical release date | 2009-12-10 |
| `revenue` | int | Worldwide box-office gross (USD) | 2,787,965,087 |
| `runtime` | float | Length in minutes | 162 |
| `spoken_languages` | JSON | Languages spoken in the film | English, Spanish |
| `status` | string | Release status | Released, Post Production, Rumored |
| `tagline` | string | Marketing tagline (sparse, ~18% missing) | "Enter the World of Pandora." |
| `title` | string | English title | Avatar |
| `vote_average` | float | Mean user rating, 0–10 *(post-release)* | 7.2 |
| `vote_count` | int | Number of user votes *(post-release)* | 11,800 |

**Engineered fields:** `roi` (revenue ÷ budget), `profit` (revenue − budget), `year`, `month`, `primary_genre`, `n_genres`, `n_production_companies`, `n_spoken_languages`, and the binary target `success`.

### Target variable

- **`success`** — binary label: `1` if `revenue ≥ 2 × budget`, else `0`. Class balance: ~56% successful / ~44% not (well balanced for classification).

### Candidate features (predictors)

To keep the model honest, only **pre-release** attributes are used as predictors — variables that would actually be known before a film opens:

- `budget` (production budget)
- `primary_genre` / genre indicators
- `runtime`
- `release month` and `year`
- `original_language`
- `n_production_companies`
- `n_spoken_languages`

**Explicitly excluded (data leakage):** `revenue`, `roi`, `profit`, `popularity`, `vote_average`, and `vote_count` are all **post-release** outcomes and are *not* used as predictors, since they would not be available at the time a prediction is needed.

---

## 4–8. (To be completed in later assignments)

Exploratory data analysis is in [`../notebooks/eda.ipynb`](../notebooks/eda.ipynb). Model training, the web application, conclusions, and references will be added in subsequent assignments.
