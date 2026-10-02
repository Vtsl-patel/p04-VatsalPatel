# P04 - Dataset Audit Pipeline

## 1. Dataset and audit requirements

I used the **New York City Airbnb Open Data (2019)** dataset from Kaggle:

`https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data`

The dataset has **48,895 rows and 16 columns**. It satisfies the four dataset checks required in R1:

| Check | Result |
|---|---|
| At least 8 columns and 1,000 rows | 48,895 rows and 16 columns |
| Numeric column with at least 3% missing | `reviews_per_month`: 10,052 missing values, about 20.6% |
| Categorical column with fewer than 10 categories | `neighbourhood_group`: 5; `room_type`: 3 |
| Categorical column with more than 30 categories | `neighbourhood`: 221 |

These checks are performed in `audit.ipynb` using `shape`, `isna()` and `nunique()`, with assertions for the required conditions.

## 2. Column audit and missing values

The audit table below records the main treatment decision for every column. The notebook calculates the audit values from the raw data.

| column | dtype | % missing | distinct | min / max or top 3 | outliers | proposed treatment |
|---|---|---:|---:|---|---:|---|
| `id` | int | 0.00 | 48,895 | 2,539 / 36,487,245 | ID, not a measurement | drop |
| `name` | object | 0.03 | high | listing titles | N/A | drop text, keep missing flag |
| `host_id` | int | 0.00 | many | 2,438 / about 274m | ID, not a measurement | drop |
| `host_name` | object | 0.04 | high | host names | N/A | drop text, keep missing flag |
| `neighbourhood_group` | object | 0.00 | 5 | Manhattan, Brooklyn, Queens | N/A | most-frequent impute + one-hot |
| `neighbourhood` | object | 0.00 | 221 | Williamsburg, Bedford-Stuyvesant, etc. | N/A | group rare values + one-hot |
| `latitude` | float | 0.00 | 19k+ | about 40.50 / 40.91 | a few IQR points | median + StandardScaler |
| `longitude` | float | 0.00 | 14k+ | about -74.24 / -73.71 | a few IQR points | median + StandardScaler |
| `room_type` | object | 0.00 | 3 | Entire home/apt, Private room, Shared room | N/A | one-hot |
| `price` | int | 0.00 | 674 | 0 / 10,000 | strong upper tail | log1p target |
| `minimum_nights` | int | 0.00 | 109 | 1 / 1,250 | strong right tail | median + log1p + RobustScaler |
| `number_of_reviews` | int | 0.00 | 394 | 0 / 629 | strong right tail | median + log1p + RobustScaler |
| `last_review` | object/date | 20.56 | about 1,765 | 2011-03-28 / 2019-07-08 | N/A | date conversion + impute + StandardScaler + missing flag |
| `reviews_per_month` | float | 20.56 | 937 | 0 / 58.5 | strong right tail | median + log1p + RobustScaler + missing flag |
| `calculated_host_listings_count` | int | 0.00 | 47 | 1 / 327 | strong right tail | median + log1p + RobustScaler |
| `availability_365` | int | 0.00 | 366 | 0 / 365 | bounded | median + MinMaxScaler |

### Missingness decisions

For `reviews_per_month`, there are 10,052 missing values. I compared `number_of_reviews` for rows where the value is missing and present. The missingness is related to observed review history, so I treat it as **MAR**. I use median imputation inside the pipeline and add `reviews_per_month_missing`.

For `last_review`, the same 10,052 rows are missing. The missing group also differs in `number_of_reviews`. I therefore treat it as **MAR**, convert the date to a numeric day value, impute inside the pipeline, and add `last_review_missing`.

For `name`, 16 values are missing (about 0.03%). For `host_name`, 21 values are missing (about 0.04%). I checked these missing rows against observed categorical columns. There was no clear pattern in the simple checks, so I treat these very small amounts as approximately **MCAR**. The raw text is dropped, but the missing indicators are kept.

For the missing columns, `audit.ipynb` prints the mean, median and standard deviation before and after imputation. The imputation is fitted within each training fold.

## 3. Numeric transformations and scaling

I did not use `StandardScaler` for every numeric feature.

`minimum_nights`, `number_of_reviews`, `reviews_per_month`, and `calculated_host_listings_count` have positive skew and long upper tails. I use `log1p` followed by `RobustScaler`. The log transformation reduces the tail and RobustScaler is less affected by extreme values.

For `latitude` and `longitude`, I use median imputation and `StandardScaler` because they are continuous location values without the same long right tail.

For `availability_365`, I use `MinMaxScaler` because the feature has a fixed 0 to 365 range.

The notebook reports skew and tail information so the scaling choices are based on the observed data rather than using one scaler for everything.

For the target `price`, I use `log1p` through `TransformedTargetRegressor` and convert predictions back with `expm1`, because the target has a strong right tail.

## 4. Categorical features and identifiers

`neighbourhood_group` has 5 categories and `room_type` has 3. I use one-hot encoding with `handle_unknown="ignore"`. These categories are not ordered, so ordinal encoding would introduce an artificial order.

There is no genuinely ordered categorical feature in this dataset, so no ordinal encoder is needed.

`neighbourhood` has 221 categories. I group rare categories using `min_frequency=20` and then one-hot encode them with `handle_unknown="ignore"`. This keeps the feature matrix smaller and also makes prediction safe when a new neighbourhood value appears.

`id` and `host_id` are identifiers, so I drop them rather than using them as normal numeric features. I also drop the raw `name` and `host_name` text because they have many unique values; their missingness indicators are retained.

## 5. Pipeline and validation

All preprocessing is kept inside one scikit-learn pipeline:

```python
Pipeline([
    ("pre", preprocessor),
    ("model", estimator)
])
```

The pipeline receives the raw DataFrame and performs imputation, transformations, scaling and encoding before fitting the estimator. This prevents preprocessing statistics from being calculated using validation folds.

`SimpleImputer(..., keep_empty_features=True)` is used where needed so a feature that is entirely missing in a training fold does not cause the pipeline to fail.

The categorical encoders use `handle_unknown="ignore"`, so unseen categories do not cause prediction errors.

The model is evaluated with 5-fold cross-validation using the complete pipeline. No preprocessing is fitted on the full dataset before cross-validation.

## 6. Leakage and unseen-category experiments

### Leakage experiment

`leak_tests.ipynb` compares the correct pipeline approach with a deliberately broken approach where preprocessing is fitted on all of `X` before cross-validation.

The broken approach is leakage because validation rows influence preprocessing values such as imputation, scaling and category grouping. The notebook prints the five scores for both approaches and their mean difference. Exact scores are not written here because they depend on running the project with the Kaggle dataset available.

### Unseen-category experiment

I create a test row containing a neighbourhood that was not present during training:

```text
NEVER_SEEN_NEIGHBOURHOOD
```

The real pipeline uses:

```python
OneHotEncoder(
    min_frequency=20,
    handle_unknown="ignore"
)
```

so prediction succeeds. A deliberately broken encoder with `handle_unknown="error"` raises a `ValueError`. This demonstrates why the unknown-category setting is part of the final pipeline.
