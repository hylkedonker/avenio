from typing import Callable

from category_encoders import CatBoostEncoder
from catboost import CatBoostClassifier
import pandas as pd
from sklearn.base import BaseEstimator
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
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from source import phenotype_features
from models import Gene2Vec, MergeRareCategories, UniqueFeatureFilter


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
mutation_columns = [
    "TP53",
    "KRAS",
    "FGFR1",
    "PTEN",
    "FBXW7",
    "KDR",
    "MTOR",
    "EGFR",
    "MET",
    "CDKN2A",
    "BRAF",
    "APC",
    "KEAP1",
    "ALK",
    "AR",
    "ERBB2",
    "NRAS",
    "NFE2L2",
    "TSC2",
    "GNAS",
    "STK11",
    "CD274",
    "CTNNB1",
    "MAP2K2",
    "IDH1",
    "NF2",
    "MAP2K1",
    "PIK3CA",
    "IDH2",
    "FLT4",
    "ESR1",
    "DDR2",
    "KIT",
    "PTCH1",
    "SMAD4",
    "SMO",
    "RNF43",
    "FGFR2",
    "JAK2",
    "CCND1",
    "GATA3",
    "PDGFRA",
]

phenotypes_to_drop = [
    "Systemischetherapie",
    # "histology_grouped",
    # "lymfmeta",
    "brainmeta",
    "adrenalmeta",
    # "livermeta",
    "stage",
    "therapyline",
    "lungmeta",
    "skeletonmeta",
]


def select_phenotype_columns(X: pd.DataFrame) -> pd.DataFrame:
    """
    Ignore mutaton data columns.
    """
    # The following list does not contain phenotypes that are not in `X`.
    phenotype_columns = [column for column in X.columns if column in phenotype_features]
    return X[phenotype_columns].copy()


def select_no_phenotype_columns(X: pd.DataFrame) -> pd.DataFrame:
    """
    Remove all phenotype columns.
    """
    # Maintain order of columns.
    no_phenotype_columns = [
        column for column in X.columns if column not in phenotype_features
    ]
    return X[no_phenotype_columns].copy()


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
    columns_to_encode = [
        column
        for column in categorical_input_columns
        if column not in phenotypes_to_drop
    ]
    category_preprocess = ColumnTransformer(
        [("LabelEncoder", OneHotEncoder(handle_unknown="ignore"), columns_to_encode)],
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
                "category_grouper",
                MergeRareCategories(
                    categorical_columns=categorical_input_columns, thresshold=30
                ),
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
            (
                "scaler",
                ColumnTransformer(
                    [("mutation_scaler", MinMaxScaler(), mutation_columns)],
                    remainder="passthrough",
                ),
            ),
            ("filter_rare_mutations", UniqueFeatureFilter(thresshold=6)),
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
            ),
            ("mutation_scaler", MinMaxScaler(), mutation_columns),
        ],
        remainder="passthrough",
    )

    # Pipeline with all features, Freeman.
    p_Freeman = Pipeline(
        steps=[
            # (
            #     "category_grouper",
            #     MergeRareCategories(
            #         categorical_columns=categorical_input_columns, thresshold=30
            #     ),
            # ),
            ("transform_columns", all_categorical_columns_transformer),
            ("classify", Estimator(**kwargs)),
        ]
    )
    return p_Freeman


def pipeline_Nikolay(Estimator, **kwargs):
    """
    Pipeline with gene embeddings.
    """
    p_Bogolyubov = Pipeline(
        steps=[
            ("vectorise_mutations", Gene2Vec(remainder="drop")),
            (
                "category_grouper",
                MergeRareCategories(
                    categorical_columns=categorical_input_columns, thresshold=30
                ),
            ),
            ("classify", Estimator(**kwargs)),
        ]
    )
    return p_Bogolyubov


def pipeline_Pyotr(Estimator, **kwargs):
    """
    Pipeline with both embeddings and categorical data.
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
    p_Kapitsa = Pipeline(
        steps=[
            ("vectorise_mutations", Gene2Vec(remainder="ignore")),
            (
                "category_grouper",
                MergeRareCategories(
                    categorical_columns=categorical_input_columns, thresshold=30
                ),
            ),
            ("transform_categories", all_categorical_columns_transformer),
            ("classify", Estimator(**kwargs)),
        ]
    )
    return p_Kapitsa


def hybrid_regressor(random_state: int = 1234) -> BaseEstimator:
    net_kwargs = {
        "random_state": random_state,
        "l1_ratio": 0.75,
        "alpha": 1.0,
        "max_iter": 1000,
    }
    tree_kwargs = {"random_state": random_state, "max_depth": 4}
    return VotingRegressor(
        estimators=[
            ("phenotype", pipeline_Richard(ElasticNet, **net_kwargs)),
            ("genetics", pipeline_Julian(DecisionTreeRegressor, **tree_kwargs)),
        ]
    )


def hybrid_classifier(random_state: int = 1234) -> BaseEstimator:
    log_kwargs = {
        "random_state": random_state,
        "penalty": "elasticnet",
        "class_weight": "balanced",
        "solver": "saga",
        "l1_ratio": 0.75,
        "C": 0.5,
    }
    tree_kwargs = {
        "criterion": "gini",
        "random_state": random_state,
        "max_depth": 5,
        "class_weight": "balanced",
    }

    return VotingClassifier(
        estimators=[
            ("phenotype", pipeline_Richard(LogisticRegression, **log_kwargs)),
            ("genetics", pipeline_Julian(DecisionTreeClassifier, **tree_kwargs)),
        ]
    )


def pipelines(Estimator, VotingEstimator=VotingClassifier, **kwargs) -> dict:
    """
    Generate pipelines for a given classifier.
    """
    d = {
        "Richard": pipeline_Richard(Estimator, **kwargs),
        "Julian": pipeline_Julian(Estimator, **kwargs),
        "Freeman": pipeline_Freeman(Estimator, **kwargs),
        # "Nikolay": pipeline_Nikolay(Estimator, **kwargs),
        # "Pyotr": pipeline_Pyotr(Estimator, **kwargs),
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
        GradientBoostingRegressor: {"random_state": random_state, "n_estimators": 15},
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
            "criterion": "gini",
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
        GaussianNB: {},
        GradientBoostingClassifier: {"random_state": random_state, "n_estimators": 15},
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
        DummyClassifier: {"strategy": "most_frequent", "random_state": random_state},
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


def calculate_pass_through_column_names_Richard():
    """
    Determine the column names that pass unaltered through the Richard pipeline.
    """
    return [
        column
        for column in phenotype_features
        if column not in categorical_input_columns
        if column not in phenotypes_to_drop
    ]


def reconstruct_categorical_variable_names_Richard(pipeline):
    """
    Determine the column names of the input columns entering Richard's classifier.
    """
    # Take the transformer right before the classifier.
    column_transformer = pipeline.steps[-2][1]
    # Consistency check: The column transformer should only contain the one hot encoder.
    assert len(column_transformer.transformers_) == 2
    # Get the column names that are transformed.
    columns = column_transformer.transformers_[0][2]
    hot_encoder = column_transformer.transformers_[0][1]
    # And generate the feature names.
    names = list(hot_encoder.get_feature_names(input_features=columns))

    # Make the names prettier.
    names = [name.replace("_", ": ") for name in names]

    # Concatenate with unaltered phenotype columns.
    names.extend(calculate_pass_through_column_names_Richard())
    return names
