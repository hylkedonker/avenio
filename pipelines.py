from typing import Callable

from category_encoders import CatBoostEncoder
from catboost import CatBoostClassifier
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, ElasticNet, LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from source import phenotype_features
from models import UniqueFeatureFilter


# From those listed above, the following columns are categorical (not counting
# the labels).
categorical_input_columns = [
    "gender",
    "stage",
    "therapyline",
    "smokingstatus",
    "Systemischetherapie",
    "histology_grouped",
    "lymfmeta",
    "brainmeta",
    "adrenalmeta",
    "livermeta",
    "lungmeta",
    "skeletonmeta",
]

phenotypes_to_drop = [
    # "Systemischetherapie",
    # "histology_grouped",
    # "lymfmeta",
    # "brainmeta",
    # "adrenalmeta",
    # "livermeta",
    # "lungmeta",
    # "skeletonmeta",
]


def select_phenotype_columns(X: pd.DataFrame) -> pd.DataFrame:
    """
    Ignore mutaton data columns.
    """
    # The following list does not contain phenotypes that are not in `X`.
    phenotype_columns = [
        column for column in X.columns if column in phenotype_features
    ]
    return X[phenotype_columns]


def select_no_phenotype_columns(X: pd.DataFrame) -> pd.DataFrame:
    """
    Remove all phenotype columns.
    """
    # Maintain order of columns.
    no_phenotype_columns = [
        column for column in X.columns if column not in phenotype_features
    ]
    return X[no_phenotype_columns]


def drop_specific_phenotypes(X: pd.DataFrame) -> pd.DataFrame:
    X_prime = X.drop(columns=phenotypes_to_drop)
    return X_prime


def pipeline_Richard(Estimator, **kwargs):
    """
    Phenotype-only pipeline Richard.
    """
    # A simple one-hot-encoder seems to work best (as opposed to more fancy
    # catboost encoder).
    # category_preprocess = CatBoostEncoder(cols=categorical_input_columns)
    category_preprocess = ColumnTransformer(
        [
            (
                "LabelEncoder",
                OneHotEncoder(handle_unknown="ignore"),
                [
                    column
                    for column in categorical_input_columns
                    if column not in phenotypes_to_drop
                ],
            )
        ],
        remainder="passthrough",
    )
    # Phenotype-only pipeline Richard.
    p_Richard = Pipeline(
        steps=[
            (
                "select_columns",
                FunctionTransformer(select_phenotype_columns, validate=False),
            ),
            (
                "remove_specific_phenotypes",
                FunctionTransformer(drop_specific_phenotypes, validate=False),
            ),
            ("transform_columns", category_preprocess),
            ("classify", Estimator(**kwargs)),
        ]
    )
    return p_Richard


def pipeline_Julian(Estimator, **kwargs):
    """
    Mutation-only pipeline Julian.
    """
    p_Julian = Pipeline(
        steps=[
            (
                "select_columns",
                FunctionTransformer(select_no_phenotype_columns, validate=False),
            ),
            ("filter_rare_mutations", UniqueFeatureFilter(thresshold=5)),
            ("classify", Estimator(**kwargs)),
        ]
    )
    return p_Julian


def pipeline_Freeman(Estimator, **kwargs):
    """
    All-feature pipeline Freeman.
    """
    all_categorical_columns_transformer = ColumnTransformer(
        [
            (
                "LabelEncoder",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_input_columns,
            )
        ],
        remainder="passthrough",
    )

    # Pipeline with all features, Freeman.
    p_Freeman = Pipeline(
        steps=[
            ("transform_columns", all_categorical_columns_transformer),
            ("classify", Estimator(**kwargs)),
        ]
    )
    return p_Freeman


def pipelines(Estimator, VotingEstimator=VotingClassifier, **kwargs) -> dict:
    """
    Generate pipelines for a given classifier.
    """
    d = {
        "Richard": pipeline_Richard(Estimator, **kwargs),
        "Julian": pipeline_Julian(Estimator, **kwargs),
        "Freeman": pipeline_Freeman(Estimator, **kwargs),
    }

    # Combine Richard & Julian into Lev.
    if VotingEstimator is not None:
        vote_kwargs = {
            "estimators": [("phenotype", d["Richard"]), ("mutation", d["Julian"])]
        }
        if type(VotingEstimator) == VotingClassifier:
            vote_kwargs["voting"] = "soft"
        p_Lev = VotingEstimator(**vote_kwargs)
        d["Lev"] = p_Lev

    return d


def build_regression_pipelines(random_state: int = 1234) -> dict:
    """
    Build a regression pipelines using a variety
    """
    regressors = {
        DecisionTreeRegressor: {"random_state": random_state, "max_depth": 4},
        RandomForestRegressor: {
            "random_state": random_state,
            "max_depth": 4,
            "n_estimators": 10,
        },
        GradientBoostingRegressor: {
            "random_state": random_state,
            "n_estimators": 15,
        },
        KNeighborsRegressor: {"n_neighbors": 5},
        ElasticNet: {
            "random_state": random_state,
            "l1_ratio": 0.75,
            "alpha": 1.0,
            "max_iter": 1000,
        },
        LinearRegression: {},
        SVR: {"kernel": "rbf", "gamma": "scale"},
        DummyRegressor: {"strategy": "median"},
    }
    return {
        str(Regressor.__name__): pipelines(
            Regressor, VotingEstimator=VotingRegressor, **kwargs
        )
        for Regressor, kwargs in regressors.items()
    }


def build_classifier_pipelines(random_state: int = 1234) -> dict:
    """
    For a variety of classifier, create a set of pipelines.
    """
    classifiers = {
        DecisionTreeClassifier: {
            "random_state": random_state,
            "max_depth": 5,
            "class_weight": "balanced",
        },
        RandomForestClassifier: {
            "random_state": random_state,
            "n_estimators": 15,
            "max_depth": 5,
            "class_weight": "balanced_subsample",
        },
        GradientBoostingClassifier: {
            "random_state": random_state,
            "n_estimators": 15,
        },
        KNeighborsClassifier: {"n_neighbors": 2, "weights": "distance"},
        LogisticRegression: {
            "random_state": random_state,
            "penalty": "elasticnet",
            "class_weight": "balanced",
            "solver": "saga",
            "l1_ratio": 0.75,
            "C": 0.5,
        },
        SVC: {
            "random_state": random_state,
            "kernel": "rbf",
            "probability": True,
            "gamma": "scale",
        },
        # CatBoostClassifier: {"random_seed": random_state},
        DummyClassifier: {
            "strategy": "most_frequent",
            "random_state": random_state,
        },
    }
    return {
        str(Classifier.__name__): pipelines(
            Classifier, VotingEstimator=VotingClassifier, **kwargs
        )
        for Classifier, kwargs in classifiers.items()
    }


def benchmark_pipelines(
    pipelines: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    metric: Callable = accuracy_score,
    **metric_kwargs,
) -> pd.DataFrame:
    """
    Make a benchmark of classifier versus preprocessing architecture.
    """
    benchmark_result = {}
    # Each classifier is associated with a set of pipelines.
    for classifier_name, classifier_pipelines in pipelines.items():
        benchmark_result[classifier_name] = {}
        # Benchmark all pipeline configurations with this classifier.
        for pipeline_name, p in classifier_pipelines.items():
            # Fit to training data.
            p.fit(X_train, y_train)
            y_train_pred = p.predict(X_train)
            y_test_pred = p.predict(X_test)
            # benchmark_result[classifier_name][f"{pipeline_name}_train"] = metric(
            #     y_train, y_train_pred, **metric_kwargs
            # )
            benchmark_result[classifier_name][f"{pipeline_name}_test"] = metric(
                y_test, y_test_pred, **metric_kwargs
            )
    return pd.DataFrame(benchmark_result).T
