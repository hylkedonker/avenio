from glob import glob
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd

from const import outcome_labels
from models import get_categorical_columns
from fragment_count.utils import load_samples_as_frame

RANDOM_STATE = 1234
np.random.seed(RANDOM_STATE)


def categorical_columns_to_lower(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all text columns to lower case.
    """
    df = data_frame.copy()
    for column in get_categorical_columns(df):
        df[column] = df[column].str.lower()

    return df


def single_out_no_mutation_patients(spread_sheet_filename: str) -> pd.Series:
    """
    Parse sheet with patient list that have no mutations.
    """
    # Read spreadsheet.
    no_mutation_data_frame = pd.read_excel(spread_sheet_filename, sheet_name=1)

    # Since the column containing the patient ids might be moved left or right, and up
    # and down, we have to find the start.
    for column in no_mutation_data_frame.columns:
        try:
            c = no_mutation_data_frame[column].str.lower()
        except AttributeError:
            # No string present in column, continue to the next one.
            continue

        if len(c[c == "patient id"]) != 0:
            j = c[c == "patient id"].index[0]
            break

    return no_mutation_data_frame[column].iloc[j + 1 :]


def load_avenio_files(
    spread_sheet_filename: str = "variant_list_20200730.xlsx",
    spss_filename: str = "clinical_20200420.sav",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load the mutation spreadsheet and SPSS phenotype data in two data frames.
    """
    # Load data from spreadsheet.
    mutation_data_frame = pd.read_excel(spread_sheet_filename, sheet_name=1)
    # Load the phenotypes from SPSS file.
    phenotypes = pd.read_spss(spss_filename)

    # And set Patient ID as index.
    phenotypes["studynumber"].name = "Patient ID"
    phenotypes.set_index(phenotypes["studynumber"].astype(int), inplace=True)
    # Convert stage from float to integer.
    columns_to_int = ["stage", "therapyline"]
    phenotypes[columns_to_int] = phenotypes[columns_to_int].astype(int)

    return (mutation_data_frame, categorical_columns_to_lower(phenotypes))


def add_clearance_patients(
    mutation_table: pd.DataFrame, clinical_sheet: pd.DataFrame
) -> pd.DataFrame:
    """
    Add mutationless patients to the mutation table by filling the rows with zeros.
    """
    # Add the patients that are not in the mutation list.
    mutationless_patients = set(clinical_sheet["studynumber"]) - set(
        mutation_table.index
    )
    # Make the difference set ordered using `tuple`, and then convert to a Series.
    mutationless_patients = pd.Series(tuple(mutationless_patients))

    no_mutations = pd.DataFrame(
        # Create table of zeros.
        np.zeros([mutationless_patients.shape[0], mutation_table.shape[1]]),
        # Use column names of `patient_mutation_frequencies`.
        columns=mutation_table.columns,
        # Index by patient id.
        index=mutationless_patients,
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
        y = X[outcome_labels].copy()
        X.drop(outcome_labels, axis=1, inplace=True)
        return (X, y)

    return (X,)


def load_fragment_statistics(
    group_by="response_grouped",
    statistic: Literal[
        "fragment_length", "fourmer", "watson_fourmer", "crick_fourmer"
    ] = "fragment_length",
    sample_type: Literal["tumor_derived", "chip", "pbmc_plus_plasma"] = "tumor_derived",
    time_point: Optional[Literal["0", "1"]] = None,
):
    """
    Load fragment statistics, seperating by clinical variable `group_by`.
    """
    _, clinical_data = load_avenio_files()

    assert group_by in clinical_data.columns
    clinical_groups = {}
    for group in clinical_data[group_by].unique():
        patients = clinical_data[group_by] == group

        time_point_suffix = "*"
        if time_point is not None:
            time_point_suffix = f"_{time_point}"

        sample_json = []
        # Collect all patient samples.
        for patient_id in clinical_data[patients].index:
            sample_json += glob(
                f"fragment_count/output/{sample_type}/"
                f"{patient_id}{time_point_suffix}.json"
            )

        clinical_groups[group] = load_samples_as_frame(
            sample_json, field_name=statistic
        )

    return clinical_groups
