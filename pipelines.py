from typing import Callable

import category_encoders as ce
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

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


def pipelines(Classifier, **kwargs) -> dict:
    """
    Generate pipelines for a given classifier.
    """

    category_preprocess = ColumnTransformer(
        [
            (
                "LabelEncoder",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_input_columns,
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
            ("transform_columns", category_preprocess),
            ("classify", Classifier(**kwargs)),
        ]
    )

    # Mutation-only pipeline Julian.
    p_Julian = Pipeline(
        steps=[
            (
                "select_columns",
                FunctionTransformer(select_no_phenotype_columns, validate=False),
            ),
            ("filter_rare_mutations", UniqueFeatureFilter(thresshold=5)),
            ("classify", Classifier(**kwargs)),
        ]
    )

    # Combine Richard & Julian into Lev.
    p_Lev = VotingClassifier(
        estimators=[("phenotype", p_Richard), ("mutation", p_Julian)],
        voting="soft",
        # Model using all the data is given less weight.
        #         weights=[2, 2, 1],
    )

    # Pipeline with all features, Freeman.
    p_Freeman = Pipeline(
        steps=[
            ("transform_columns", category_preprocess),
            ("classify", Classifier(**kwargs)),
        ]
    )

    return {
        "Richard": p_Richard,
        "Julian": p_Julian,
        "Lev": p_Lev,
        "Freeman": p_Freeman,
    }


def build_pipelines(random_state: int = 1234) -> dict:
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
        DummyClassifier: {
            "strategy": "most_frequent",
            "random_state": random_state,
        },
    }
    return {
        str(Classifier.__name__): pipelines(Classifier, **kwargs)
        for Classifier, kwargs in classifiers.items()
    }


def benchmark_pipelines(
    pipelines: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    metric: Callable = accuracy_score,
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
            benchmark_result[classifier_name][f"{pipeline_name}_train"] = metric(
                y_train, y_train_pred
            )
            benchmark_result[classifier_name][f"{pipeline_name}_test"] = metric(
                y_test, y_test_pred
            )
    return pd.DataFrame(benchmark_result).T
