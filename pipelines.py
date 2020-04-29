from typing import Callable

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, NMF
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    LogisticRegression,
    ElasticNet,
    LinearRegression,
)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer, OneHotEncoder
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np

from const import categorical_phenotypes as categorical_input_columns
from const import clinical_features
from models import (
    AggregateColumns,
    AutoMaxScaler,
    SparseFeatureFilter,
)


RANDOM_STATE = 1234

mutation_columns = [
    "KRAS",
    "PIK3R1",
    "PMS2",
    "SMAD4",
    "BRAF",
    "CCND1",
    "FLT4",
    "KDR",
    "TERT",
    "TP53",
    "BRCA1",
    "CDK4",
    "EGFR",
    "KEAP1",
    "MET",
    "PDGFRA",
    "RET",
    "TSC2",
    "ERBB2",
    "FGFR1",
    "PIK3CA",
    "PTEN",
    "RB1",
    "CDKN2A",
    "FLT1",
    "AR",
    "ESR1",
    "MTOR",
    "STK11",
    "FBXW7",
    "FLT3",
    "ABL1",
    "RNF43",
    "ROS1",
    "ALK",
    "APC",
    "MSH2",
    "BRCA2",
    "MAP2K1",
    "SMO",
    "FGFR3",
    "TSC1",
    "CCND2",
    "PTCH1",
    "FGFR2",
    "MLH1",
    "KIT",
    "VHL",
    "MSH6",
    "RAF1",
    "NTRK1",
    "CTNNB1",
    "PDCD1LG2",
    "CDK6",
    "NF2",
    "JAK3",
    "NRAS",
    "NFE2L2",
    "AKT1",
    "CD274",
    "MAP2K2",
    "IDH1",
    "JAK2",
    "PDGFRB",
    "IDH2",
    "GNAS",
    "CSF1R",
    "ARAF",
    "DDR2",
    "GATA3",
    "GNA11",
    "GNAQ",
]
phenotypes_to_drop = [
    "Systemischetherapie",
    # # "histology_grouped",
    # # "lymfmeta",
    # "brainmeta",
    # "adrenalmeta",
    # # "livermeta",
    "stage",
    # "therapyline",
    # "lungmeta",
    # "skeletonmeta",
]
meta_columns = [
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
    phenotype_columns = [column for column in X.columns if column in clinical_features]
    return X[phenotype_columns].copy()


def select_no_phenotype_columns(X: pd.DataFrame) -> pd.DataFrame:
    """
    Remove all phenotype columns.
    """
    # Maintain order of columns.
    no_phenotype_columns = [
        column for column in X.columns if column not in clinical_features
    ]
    return X[no_phenotype_columns].copy()


def drop_specific_phenotypes(X: pd.DataFrame) -> pd.DataFrame:
    X_prime = X.drop(columns=phenotypes_to_drop)
    return X_prime.copy()


def metastases_columns_to_TNM(X: pd.DataFrame) -> pd.DataFrame:
    """
    Group the metastases columns according to TNM staging system.
    """
    X_prime = X.copy()

    # Indicate presence of lymphnode metastasis (the primary metastasis is always
    # present for this data set).
    d = {"no metastasis present": "absent", "metastasis present": "present"}
    X_prime["TNM-N"] = X_prime["lymfmeta"].replace(d)

    # Count the number of distant metasteses regions.
    X_prime["TNM-M_count"] = (
        X_prime[["brainmeta", "adrenalmeta", "livermeta", "skeletonmeta"]]
        .replace({"no metastasis present": 0, "metastasis present": 1})
        .sum(axis=1)
    )

    return X_prime.drop(columns=meta_columns)


def clinical_data_curation(X: pd.DataFrame) -> pd.DataFrame:
    """
    Further curation of the clinical data.
    """
    X_prime = X.copy()

    # After inspection the other histologies are partially adeno.
    X_prime["histology_grouped"].replace({"other": "adeno"}, inplace=True)
    first_line_therapy = X_prime["therapyline"].isin([0, 1])

    # According to the selection criterea of the study, 0th and 1st line should be
    # combined.
    X_prime.loc[first_line_therapy, "therapyline"] = "0+1"
    X_prime.loc[~first_line_therapy, "therapyline"] = ">1"

    # Unknown smoking status are probably smokers according Harry.
    X_prime["smokingstatus"].replace({"unknown": "smoker"}, inplace=True)
    # Group together current and previous smokers.
    X_prime["smokingstatus"].replace(
        {"previous": "current+previous", "smoker": "current+previous"}, inplace=True
    )

    d = {"no metastasis present": "absent", "metastasis present": "present"}
    X_prime[meta_columns] = X_prime[meta_columns].replace(d)

    # Partition age in two.
    young = X_prime["Age"] < 65
    X_prime["age"] = r"$\geq$ 65"
    X_prime.loc[young, "age"] = "<65"
    # Remove original column.
    X_prime.drop(columns="Age", inplace=True)

    # All clinical variables are categories.
    new_clinical_features = clinical_features.copy()
    new_clinical_features[new_clinical_features.index('Age')] = 'age'
    X_prime[new_clinical_features] = X_prime[new_clinical_features].astype('category')
    return X_prime


def clinical_preprocessing_steps() -> list:
    """
    Standard pipeline preprocessing steps for the clinical data.
    """
    return [
        (
            "clinical_curation",
            FunctionTransformer(clinical_data_curation, validate=False),
        ),
        (
            "filter_clinical_variables",
            FunctionTransformer(drop_specific_phenotypes, validate=False),
        ),
    ]


def clinical_encoder_step():
    """
    Encode the clinical categories as numbers.
    """
    columns_to_encode = [
        column
        for column in categorical_input_columns
        if column not in phenotypes_to_drop
    ]
    columns_to_encode.append("age")

    return ColumnTransformer(
        [("LabelEncoder", OneHotEncoder(handle_unknown="ignore"), columns_to_encode)],
        remainder="passthrough",
    )


def pipeline_Richard(Estimator, **kwargs):
    """
    Phenotype-only pipeline Richard.
    """
    steps_Richard = [
        (
            "select_clinical_data",
            FunctionTransformer(select_phenotype_columns, validate=False),
        ),
        *clinical_preprocessing_steps(),
        ("encode_clinical_categories", clinical_encoder_step()),
        ("estimator", Estimator(**kwargs)),
    ]

    return Pipeline(steps=steps_Richard)


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
            ("filter_rare_mutations", SparseFeatureFilter(top_k_features=6)),
            ("estimator", Estimator(**kwargs)),
        ]
    )
    return p_Julian


def pipeline_Freeman(Estimator, **kwargs):
    """
    Pipeline with clinical + genomic data: Freeman.
    """
    steps_Freeman = [
        *clinical_preprocessing_steps(),
        ("normalise_genomic_data", AutoMaxScaler(ignore_columns=["Age"])),
        ("encode_clinical_categories", clinical_encoder_step()),
        ("estimator", Estimator(**kwargs)),
    ]

    return Pipeline(steps=steps_Freeman)


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
            "alpha": 2,
            "max_iter": 1000,
        },
        LinearRegression: {},
        ARDRegression: {},
        BayesianRidge: {},
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
            "min_samples_leaf": 1,
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
            # Choose l2 norm in combination with newton conjugate gradient solver for
            # converging results (elasticnet is slightly better, but not always
            # converges correctly).
            "solver": "newton-cg",
            "penalty": "l2",
            # ####### FOR ELASTIC NET#####
            # "solver": "saga",
            # "penalty": "elasticnet",
            # "l1_ratio": 0.025,
            # ############################
            "class_weight": "balanced",
            "multi_class": "auto",
            # Make an unbiased model.
            "fit_intercept": False,
            "C": 1.0,
            "max_iter": 5000,
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
            Classifier,
            # VotingEstimator=VotingClassifier,
            VotingEstimator=None,
            **kwargs,
        )
        for Classifier, kwargs in classifiers.items()
    }


def benchmark_pipelines(
    pipelines: dict, X: pd.DataFrame, y: pd.Series, metric: Callable = accuracy_score
) -> pd.DataFrame:
    """
    Make a benchmark of classifier versus preprocessing architecture.
    """
    benchmark_result = {}
    # Each classifier is associated with a set of pipelines.
    for classifier_name, classifier_pipelines in pipelines.items():
        classifier_scores = {}
        benchmark_result[classifier_name] = classifier_scores
        # Benchmark all pipeline configurations with this classifier.
        for pipeline_name, p in classifier_pipelines.items():
            k_fold_scores = cross_val_score(p, X, y, scoring=metric, cv=5)
            classifier_scores[f"{pipeline_name} mean"] = np.mean(k_fold_scores)
            classifier_scores[f"{pipeline_name} std"] = np.std(k_fold_scores)

    return pd.DataFrame(benchmark_result).T


def calculate_pass_through_column_names_Richard(pipeline):
    """
    Determine the column names that pass unaltered through the Richard pipeline.
    """
    columns = [
        column
        for column in clinical_features
        if column not in categorical_input_columns
        if column not in phenotypes_to_drop
    ]
    # Remove age column, if necessary.
    column_transformer = pipeline.steps[-2][1]
    if "age_discretizer" in column_transformer.named_transformers_:
        columns.remove("Age")
    return columns


def calculate_pass_through_column_names_Freeman(pipeline):
    """
    Determine the column names that pass unaltered through the Freeman pipeline.
    """
    return pipeline.named_steps["normalise_genomic_data"].columns_to_transform_


def reconstruct_categorical_variable_names(pipeline):
    """
    Determine the column names of the categorical (clinical) input columns.
    """
    # Take the transformer right before the classifier.
    column_transformer = pipeline.steps[-2][1]

    # Get the column names that are transformed.
    # 1) One-hot-encoder.
    columns = column_transformer.transformers_[0][2]
    hot_encoder = column_transformer.transformers_[0][1]
    # And generate the feature names.
    names = list(hot_encoder.get_feature_names(input_features=columns))

    # 2) Discretizer, if available.
    if "age_discretizer" in column_transformer.named_transformers_:
        age_binner = column_transformer.named_transformers_["age_discretizer"]
        edges = age_binner.bin_edges_[
            0
        ]  # Not sure why this is a tuple, with 1 element.

        # Generate labels for the bins.
        age_labels = []
        for i in range(len(edges) - 1):
            age_labels.append("{:.0f}<=age<{:.0f}".format(edges[i], edges[i + 1]))
        # And add age bin labels.
        names.extend(age_labels)

    # Make the names prettier.
    new_names = {
        "histology_grouped": "histology",
        "smokingstatus": "smoker",
        "therapyline": "therapy line",
    }
    # Make replacement for the keys above.
    for i, name in enumerate(names):
        for key, value in new_names.items():
            if key in name:
                names[i] = name.replace(key, value)
                continue

    names = [name.replace("_", ": ") for name in names]
    return names


def evaluate_training_size_dependence(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    metric: Callable = accuracy_score,
    **metric_kwargs,
):
    """
    Calculate model performance as a function of training size.
    """
    # Store training size and respective score in these variables.
    sizes = []
    scores = []

    k = 5  # K-fold cross validation.

    # k-fold cross validation of training size dependence.
    # Keep track of scores for this particular fold.
    for train, test in KFold(n_splits=k).split(X):
        if isinstance(X, pd.DataFrame):
            X_train, X_test, y_train, y_test = (
                X.iloc[train],
                X.iloc[test],
                y.iloc[train],
                y.iloc[test],
            )
        else:
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

        m = X_train.shape[0]

        # Increase training size in multiples of `r`.
        # 10 = m/r^n ==> n ln r = ln [m/10]
        r = 1.5
        n_max = int(np.floor(np.log(m / 10) / np.log(r)))

        fold_scores = []
        fold_sizes = []
        # We require `m_i` (number of records) to be at least 10.
        for i in range(n_max):
            n = n_max - i - 1
            m_i = int(np.floor(m / r ** n))
            # Save size.
            fold_sizes.append(m_i)

            # Train model for reduced data set, of this particular fold.
            p = pipeline
            X_train_slice, y_train_slice = X_train.iloc[:m_i], y_train.iloc[:m_i]
            p.fit(X_train_slice, y_train_slice)

            # Calculate and store metric on test set.
            y_test_pred = p.predict(X_test)
            fold_scores.append(metric(y_test, y_test_pred, **metric_kwargs))

        sizes.append(fold_sizes)
        scores.append(fold_scores)

    # Calculate mean and standard deviation over folds.
    sizes, scores = np.array(sizes), np.array(scores)
    return np.mean(sizes, axis=0), np.mean(scores, axis=0), np.std(scores, axis=0)
