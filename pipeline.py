"""Scikit-learn pipeline for the NYC Airbnb dataset."""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler, FunctionTransformer


class DateToDays(BaseEstimator, TransformerMixin):
    """Convert last_review dates to days from the first date seen during fit."""

    def fit(self, X, y=None):
        dates = pd.to_datetime(pd.DataFrame(X).iloc[:, 0], errors="coerce")
        self.origin_ = dates.min()
        if pd.isna(self.origin_):
            self.origin_ = pd.Timestamp("2011-01-01")
        return self

    def transform(self, X):
        dates = pd.to_datetime(pd.DataFrame(X).iloc[:, 0], errors="coerce")
        days = (dates - self.origin_).dt.days
        return days.to_numpy().reshape(-1, 1).astype(float)


def _log1p(X):
    return np.log1p(np.asarray(X, dtype=float))


def build_pipeline():
    log_numeric = [
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
    ]
    standard_numeric = ["latitude", "longitude"]
    availability_numeric = ["availability_365"]

    numeric = [
        (
            "standard",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
            ]),
            standard_numeric,
        ),
        (
            "log_robust",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("log", FunctionTransformer(_log1p, feature_names_out="one-to-one")),
                ("scaler", RobustScaler()),
            ]),
            log_numeric,
        ),
        (
            "availability",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", MinMaxScaler()),
            ]),
            availability_numeric,
        ),
    ]

    last_review = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="2011-01-01", keep_empty_features=True)),
        ("date_to_days", DateToDays()),
        ("scaler", StandardScaler()),
    ])

    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=True)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            *numeric,
            ("last_review", last_review, ["last_review"]),
            (
                "missing_indicators",
                MissingIndicator(features="all"),
                ["name", "host_name", "last_review", "reviews_per_month"],
            ),
            ("categorical", categorical, ["neighbourhood_group", "neighbourhood", "room_type"]),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    return Pipeline([
        ("pre", preprocessor),
        (
            "model",
            TransformedTargetRegressor(
                regressor=Ridge(alpha=10.0),
                func=np.log1p,
                inverse_func=np.expm1,
            ),
        ),
    ])
