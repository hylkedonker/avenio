from typing import Callable, List, Tuple

import category_encoders as ce
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    OrdinalEncoder,
    OneHotEncoder,
    LabelBinarizer,
)
from sklearn.tree import DecisionTreeClassifier

from models import UniqueFeatureFilter


# Phenotype features that serve as input for the model.
phenotype_features = [
    "gender",
    "leeftijd",
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

# Phenotype labels that we wish to predict.
phenotype_labels = [
    "Clinical_Response",
    "response_grouped",
    "progressie",
    "PFS_days",
    "OS_days",
    "OS_months",
    "PFS_months",
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
    }
    return {
        str(Classifier): pipelines(Classifier, **kwargs)
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
            benchmark_result[classifier_name][f"{pipeline_name}_ttest"] = metric(
                y_test, y_test_pred
            )
    return pd.DataFrame(benchmark_result)


from source import add_mutationless_patients, load_avenio_files
from transform import (
    clean_mutation_columns,
    dummy_encode_mutations,
    get_top_correlated,
    mutation_train_test_split,
    patient_allele_frequencies,
    get_top_genes,
)


# Load data from spreadsheet and SPSS files.
mutation_data_frame, no_mutation_found_patients, phenotypes = load_avenio_files()

# Vocabulary is the entire dataset, not only training set. Otherwise we run into
# problems during inference.
gene_vocabulary = mutation_data_frame["Gene"].unique()
# allele_columns = ["T0: Allele \nFraction", "T1: Allele Fraction"]
allele_columns = [
    "T0: No. Mutant \nMolecules per mL",
    "T1: No. Mutant \nMolecules per mL",
]

# Convert particular columns to numbers and drop rows with missing data.
mutation_data_frame = clean_mutation_columns(mutation_data_frame)

patient_mutation_frequencies = patient_allele_frequencies(
    mutation_data_frame, gene_vocabulary
)
patient_mutation_frequencies = add_mutationless_patients(
    patient_mutation_frequencies, no_mutation_found_patients
)
phenotypes_to_keep = phenotype_features + phenotype_labels
# Combine mutation data and phenotype data.
X = pd.merge(
    left=patient_mutation_frequencies,
    right=phenotypes[phenotypes_to_keep],
    left_index=True,
    right_index=True,
)

X.dropna(subset=["response_grouped"], inplace=True)
import ipdb

ipdb.set_trace()
# Extract the labels for the classifier.
y_resp = X.pop("Clinical_Response")
y_resp_gp = X.pop("response_grouped")
y_prog = X.pop("progressie")
