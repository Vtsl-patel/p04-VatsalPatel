# NYC Airbnb Dataset — Dataset Audit Pipeline

Dataset audit and leakage-safe preprocessing using the **New York City Airbnb Open Data** dataset from Kaggle.

The project checks data quality, missingness, scaling, categorical encoding, and builds a leakage-safe machine-learning pipeline.

---

## Dataset Source

The dataset is:

**New York City Airbnb Open Data**

- **Platform:** Kaggle
- **Dataset:** `dgomonov/new-york-city-airbnb-open-data`

Dataset source:

https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data

---

## Files

- `audit.ipynb` — dataset audit and preprocessing decisions
- `leak_tests.ipynb` — pipeline and leakage tests
- `pipeline.py` — production preprocessing and modelling pipeline
- `justification.md` — project justification

---

# How to Run

Upload `audit.ipynb` to **Google Colab** and run it from top to bottom.

Keep `pipeline.py` in the same folder and upload it to Colab before running the notebooks.

Then upload `leak_tests.ipynb` and run it from top to bottom.
