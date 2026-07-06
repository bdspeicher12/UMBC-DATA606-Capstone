# Data

This folder contains the dataset used in the capstone project.

## Contents

- **`tmdb_5000_movies.csv`** — the TMDB 5000 Movie Dataset (The Movie Database, via Kaggle).

## Dataset summary

| Property | Value |
| --- | --- |
| Rows (movies) | 4,803 (≈3,228 usable after cleaning) |
| Columns | 20 |
| File size | ~5.7 MB |
| Time period | Release years 1916–2016 |
| Unit of a row | One movie |

Key fields include `budget`, `revenue`, `genres`, `runtime`, `release_date`, `original_language`, and `production_companies`. A full data dictionary, the cleaning steps, and the engineered features (including the `success` target and the timing features) are documented in [`../docs/report.md`](../docs/report.md).
