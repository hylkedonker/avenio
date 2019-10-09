from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def dummy_encode_mutations(
    data_frame: pd.DataFrame, gene_vocabulary: Iterable
) -> pd.DataFrame:
    """
    Dummy encode mutations for each patient.
    """
    # Create an empty data frame first.
    dummy_data_frame = pd.DataFrame(
        0,
        # Rows are patients
        index=data_frame["Patient ID"].unique(),
        # Columns are the number of mutations per gene.
        columns=gene_vocabulary,
    )

    # Fill all columns with the number of occurences of each gene per patient.
    for multi_index, patient_genes in data_frame.groupby(["Patient ID", "Gene"]):
        # Unpack patient_id and the gene from the `MultiIndex`.
        patient_id, gene = multi_index
        # Store number of mutations of this particular gene.
        dummy_data_frame.loc[patient_id][gene] += len(patient_genes)

    return dummy_data_frame


def patient_allele_frequencies(
    data_frame: pd.DataFrame,
    gene_vocabulary: Iterable,
    transformation: Callable = lambda x, y: y - x,
    allele_freq_columns: List[str] = [
        "T0: Allele \nFraction",
        "T1: Allele Fraction",
    ],
) -> pd.DataFrame:
    """
    For each patient, calculate allele frequency increase (by default) of
    mutations.

    For each mutation for a given patient, calculate `transformation(f_t0,
    f_t1)` from the allele frequencies measured as t0 and t1.
    """
    # The transformation is calculated between two columns (t0 and t1).
    if len(allele_freq_columns) != 2:
        raise ValueError("Allele frequency columns must be precisely two!")

    # The columns that were passed must actually exist.
    column_names = data_frame.columns
    if (
        allele_freq_columns[0] not in column_names
        or allele_freq_columns[1] not in column_names
    ):
        raise KeyError(
            "Column lookup error in `allel_freq_columns` = {}.".format(
                allele_freq_columns
            )
        )

    # There may not be any NA values in the columns.
    if (
        sum(data_frame[allele_freq_columns[0]].isna()) > 0
        or sum(data_frame[allele_freq_columns[1]].isna()) > 0
    ):
        raise ValueError("NA values found in allele frequency columns!")

    # Create an empty data frame first.
    patient_data_frame = pd.DataFrame(
        0.0,
        # Rows are patients
        index=data_frame["Patient ID"].unique(),
        # Columns are the calculated allele frequency differences (unless
        # specified otherwise with `transformation`) per gene mutation.
        columns=gene_vocabulary,
    )

    # Go through all the mutations.
    for index, row in data_frame.iterrows():
        patient_id = row["Patient ID"]
        gene = row["Gene"]
        # Extract allele frequencies at time t0 and t1.
        f_t0, f_t1 = row[allele_freq_columns]

        # Carry out the transformation on the two allele frequencies (by default
        # difference), and store result in the corresponding gene column for the
        # given patient.
        patient_data_frame.loc[patient_id, gene] = transformation(f_t0, f_t1)

    return patient_data_frame


def mutation_train_test_split(
    mutation_data_frame: pd.DataFrame, test_fraction: float = 0.3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the mutation data frame in a train and test set.
    """
    patients = mutation_data_frame["Patient ID"].unique()
    np.random.shuffle(patients)

    # Determine the largest element no longer in the training set.
    cut_off = int(np.ceil(len(patients) * (1 - test_fraction)))
    train_patients = patients[:cut_off]

    # Put records in train/validation set according to "Patient ID".
    train_rows = mutation_data_frame["Patient ID"].isin(train_patients)
    train_data_frame = mutation_data_frame.loc[train_rows]
    test_data_frame = mutation_data_frame[~train_rows]

    return (train_data_frame, test_data_frame)


def get_top_correlated(
    correlation_data_frame: pd.DataFrame,
    ascending: bool = False,
    top_count: int = 15,
    gene_counts: Optional = None,
) -> pd.DataFrame:
    """
    Get the top correlated genes in ascending (descending) order.
    """
    # Get the maximal cell by:
    # 1) flatting array.
    corr_flat = correlation_data_frame.values.flatten()

    # 2) and sorting indices.
    if not ascending:
        top_indices = np.argsort(corr_flat)
    else:
        top_indices = np.argsort(corr_flat)

    # 3) Calculating indices back to original dataframe.
    i, j = np.unravel_index(top_indices, correlation_data_frame.shape)

    # Finally store results in data frame.
    columns = {
        "gene 1": correlation_data_frame.index[i],
        "gene 2": correlation_data_frame.index[j],
        "correlation": corr_flat[top_indices],
    }
    df = pd.DataFrame(columns)

    # Add columns with gene counts if passed.
    if gene_counts is not None:
        df["# gene 1"] = gene_counts[correlation_data_frame.index[i]].values
        df["# gene 2"] = gene_counts[correlation_data_frame.index[j]].values

    # Remove diagonal elements (i, i).
    diagonal_genes = df["gene 1"] == df["gene 2"]
    df = df[~diagonal_genes]
    # Remove permutations (i, j) ~ (j, i).
    even_indices = df.index % 2 == 0
    df = df[even_indices]

    # Sort and truncate size.
    return df.sort_values([df.columns[2]], ascending=ascending).iloc[:top_count]
