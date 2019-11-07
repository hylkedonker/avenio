from typing import Tuple

import numpy as np
import pandas as pd

from utils import get_categorical_columns


RANDOM_STATE = 1234
np.random.seed(RANDOM_STATE)


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


def categorical_columns_to_lower(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all text columns to lower case.
    """
    df = data_frame.copy()
    for column in get_categorical_columns(df):
        df[column] = df[column].str.lower()

    return df


def load_avenio_files(
    spread_sheet_filename: str = "2019-08-27_PLASMA_DEFAULT_Results_Groningen.xlsx",
    spss_filename: str = "phenotypes_20191018.sav",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load the mutation spreadsheet and SPSS phenotype data in two data frames.
    """
    # Load data from spreadsheet.
    mutation_data_frame = pd.read_excel(spread_sheet_filename, sheet_name=2)
    # Load the phenotypes from SPSS file.
    phenotypes = pd.read_spss(spss_filename)

    # List of patients with no mutation.
    no_mutation_found_patients = (
        pd.read_excel(spread_sheet_filename, sheet_name=1).dropna().iloc[:, 2]
    )
    # Identify patients for which with missing sequencing data.
    no_mutation_patient_spss = phenotypes[phenotypes["VAR00001"].isna()]["studynumber"]
    # The mutation spreadsheet doesn't have all the data yet, so it must be a
    # subset of all the patients.
    assert set(no_mutation_found_patients).issubset(no_mutation_patient_spss)

    # And set Patient ID as index.
    phenotypes["studynumber"].name = "Patient ID"
    phenotypes.set_index(phenotypes["studynumber"].astype(int), inplace=True)
    # Convert stage from float to integer.
    columns_to_int = ["stage", "therapyline"]
    phenotypes[columns_to_int] = phenotypes[columns_to_int].astype(int)

    return (
        mutation_data_frame,
        no_mutation_found_patients,
        categorical_columns_to_lower(phenotypes),
    )


def add_mutationless_patients(
    mutation_table: pd.DataFrame, mutationless_patients: pd.Series
) -> pd.DataFrame:
    """
    Add mutationless patients to the mutation table by filling the rows with zeros.
    """
    no_mutations = pd.DataFrame(
        # Create table of zeros.
        np.zeros([mutationless_patients.shape[0], mutation_table.shape[1]]),
        # Use column names of `patient_mutation_frequencies`.
        columns=mutation_table.columns,
        # Index by patient id.
        index=mutationless_patients.values,
    )
    # Append to table with patient mutations.
    return mutation_table.append(no_mutations)


def read_preprocessed_data(filename: str, split_label=True) -> Tuple[pd.DataFrame]:
    """
    Read preprocessed data from disk and seperate features from labels if needed.
    """
    X = pd.read_csv(filename, sep="\t")
    X.set_index(X.columns[0], drop=True, inplace=True)
    X.index.name = "Patient ID"
    if split_label:
        y = X[phenotype_labels].copy()
        X.drop(phenotype_labels, axis=1, inplace=True)
        return (X, y)

    return (X,)
