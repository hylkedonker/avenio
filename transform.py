from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy as sp
from sklearn.model_selection import train_test_split

from source import (
    add_mutationless_patients,
    load_avenio_files,
    phenotype_features,
    phenotype_labels,
)


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
    allele_columns: List[str] = ["T0: Allele \nFraction", "T1: Allele Fraction"],
    handle_duplicates="sum",
) -> pd.DataFrame:
    """
    For each patient, calculate allele frequency increase (by default) of
    mutations.

    For each mutation for a given patient, calculate `transformation(f_t0,
    f_t1)` from the allele frequencies measured as t0 and t1.
    """
    allowed_duplicate_actions = ("min", "max", "ignore", "concat", "sum")
    if handle_duplicates.lower() not in allowed_duplicate_actions:
        raise ValueError(
            "Allowed values for handle_duplicates are {}.".format(
                allowed_duplicate_actions
            )
        )

    # The transformation is calculated between two columns (t0 and t1).
    if len(allele_columns) != 2:
        raise ValueError("Allele frequency columns must be precisely two!")

    # The columns that were passed must actually exist.
    column_names = data_frame.columns
    if allele_columns[0] not in column_names or allele_columns[1] not in column_names:
        raise KeyError(
            "Column lookup error in `allel_freq_columns` = {}.".format(allele_columns)
        )

    # There may not be any NA values in the columns.
    if (
        sum(data_frame[allele_columns[0]].isna()) > 0
        or sum(data_frame[allele_columns[1]].isna()) > 0
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

    def first_element(x):
        return x.iloc[0]

    # Default is ignore, taking the first element that is found.
    select_from_duplicates = first_element
    if handle_duplicates == "min":
        select_from_duplicates = np.min
    elif handle_duplicates == "max":
        select_from_duplicates = np.max
    elif handle_duplicates == "sum":
        select_from_duplicates = np.sum

    # Go through all the mutations.
    for group_index, grouped in data_frame.groupby(["Patient ID", "Gene"]):
        patient_id, gene = group_index

        # Extract allele frequencies at time t0 and t1.
        f_t0, f_t1 = (grouped[allele_columns[0]], grouped[allele_columns[1]])

        # Carry out the transformation on the two allele frequencies (by default
        # difference), and store result in the corresponding gene column for the
        # given patient.
        f = transformation(f_t0, f_t1)
        patient_data_frame.loc[patient_id, gene] = select_from_duplicates(f)

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
    correlations: pd.DataFrame,
    correlation_pvalue: Optional[pd.DataFrame] = None,
    ascending: bool = False,
    top_count: int = 15,
    gene_counts: Optional = None,
) -> pd.DataFrame:
    """
    Get the top correlated genes in ascending (descending) order.
    """
    # Get the maximal cell by:
    # 1) flatting array.
    corr_flat = correlations.values.flatten()

    # 2) and sorting indices.
    if not ascending:
        top_indices = np.argsort(corr_flat)
    else:
        top_indices = np.argsort(corr_flat)

    # 3) Calculating indices back to original dataframe.
    i, j = np.unravel_index(top_indices, correlations.shape)

    # Finally store results in data frame.
    columns = {
        "gene 1": correlations.index[i],
        "gene 2": correlations.index[j],
        "correlation": corr_flat[top_indices],
    }

    if correlation_pvalue is not None:
        pval_flat = correlation_pvalue.values.flatten()
        columns["p-value"] = pval_flat[top_indices]

    df = pd.DataFrame(columns)

    # Add columns with gene counts if passed.
    if gene_counts is not None:
        df["# gene 1"] = gene_counts[correlations.index[i]].values
        df["# gene 2"] = gene_counts[correlations.index[j]].values

    # Remove diagonal elements (i, i).
    diagonal_genes = df["gene 1"] == df["gene 2"]
    df = df[~diagonal_genes]
    # Remove permutations (i, j) ~ (j, i).
    even_indices = df.index % 2 == 0
    df = df[even_indices]

    # Sort and truncate size.
    return df.sort_values([df.columns[2]], ascending=ascending).iloc[:top_count]


def clean_mutation_columns(
    input_data: pd.DataFrame,
    columns_to_number=[
        "T0: Allele \nFraction",
        "T1: Allele Fraction",
        "T0: No. Mutant \nMolecules per mL",
        "T1: No. Mutant \nMolecules per mL",
    ],
    fill_ND=np.nan,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Seperate data in a clean, converted set, and the remainder, cotaining missing
    values.

    Returns:
        (pd.DataFrame, pd.DataFrame): first element is the clean data, and the
            second elemnt contains data with missing values.
    """
    clean_data = input_data.copy()

    def convert_percentage_to_float(input_element):
        """ Convert to float and divide by 100 if it contains the "%" character. """
        if isinstance(input_element, str) and r"%" in input_element:
            return float(input_element.rstrip(r"%")) / 100.0
        return input_element

    # 1) Replace ND values with NA.
    clean_data[columns_to_number] = clean_data.loc[:, columns_to_number].replace(
        "ND", fill_ND
    )

    for column_name in columns_to_number:
        # 2) Percentage to float.
        column_data = clean_data[column_name].apply(convert_percentage_to_float)
        # 3) Convert to float.
        clean_data.loc[:, column_name] = pd.to_numeric(column_data, errors="coerce")

    # 4) Split data in clean rows, and rows with NA values.
    na_rows = clean_data[columns_to_number].isna().any(axis=1)
    return (clean_data[~na_rows], clean_data[na_rows])


def get_top_genes(data_frame: pd.DataFrame, thresshold: int = 5) -> np.ndarray:
    """
    Thresshold: genes must occur at least this many times.
    """
    # What elements are non-zero?
    non_zero_values = data_frame[data_frame != 0]
    # Pick columns that have at least `LAMBDA` occurences.
    gene_thressholded = non_zero_values.count() >= thresshold
    genes_to_pick = gene_thressholded.index[gene_thressholded]
    frequent_mutations = genes_to_pick.values
    return frequent_mutations


def load_process_and_store_spreadsheets(
    spread_sheet_filename: str = "2019-02-12_FINAL_RESULTS_SomaticAll.xlsx",
    spss_filename: str = "phenotypes_20191018.sav",
    transformation: Callable = lambda x, y: y - x,
    allele_columns: List[str] = ["T0: Allele \nFraction", "T1: Allele Fraction"],
    random_state: int = 1234,
    all_filename: str = "output/all_data.tsv",
    train_filename: str = "output/train.tsv",
    test_filename: str = "output/test.tsv",
):
    """
    Read, clean, transform and store raw data.

    1) Load the mutation Excel spreadsheet and the phenotype SPSS file.
    2) Transform the mutation columns.
    3) Combine mutation and phenotype data.
    4) Split and store data to disk.
    """
    # Load data from spreadsheet and SPSS files.
    patient_mutations, patient_no_mutations, phenotypes = load_avenio_files(
        spread_sheet_filename, spss_filename
    )

    # Vocabulary is the entire dataset, not only training set. Otherwise we run
    # into problems during inference.
    gene_vocabulary = patient_mutations["Gene"].unique()

    # Convert particular columns to numbers and drop rows with missing data.
    clean_patient_mutations, dirty_patient_mutations = clean_mutation_columns(
        patient_mutations,
        columns_to_number=allele_columns,
        # fill_ND=0.0
    )

    clean_patients = clean_patient_mutations["Patient ID"].unique()
    dirty_patients = dirty_patient_mutations["Patient ID"].unique()

    # Verify that the combination of the patients must give all patients.
    assert set(clean_patients).union(set(dirty_patients)) == set(
        patient_mutations["Patient ID"].unique()
    )

    # # Verify that all of the patients are in the `clean_patients` records.
    # assert set(dirty_patients).issubset(set(clean_patients))

    # Verify that there are no more NA values.
    assert (
        clean_patient_mutations[allele_columns].dropna().shape[0]
        == clean_patient_mutations[allele_columns].shape[0]
    )
    # And that everything went in to the "dirty" records.
    assert dirty_patient_mutations[allele_columns].dropna().shape[0] == 0

    # Perform `transformation` on `allele_columns`. That is, calculate change in
    # DNA mutation.
    patient_mutation_frequencies = patient_allele_frequencies(
        clean_patient_mutations,
        gene_vocabulary,
        allele_columns=allele_columns,
        transformation=transformation,
    )

    # Don't forget about patient for which no mutations where found.
    patient_mutation_frequencies = add_mutationless_patients(
        patient_mutation_frequencies, patient_no_mutations
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

    f_test = 0.3
    X_train, X_test = train_test_split(X, test_size=f_test, random_state=random_state)
    # Write data to disk.
    X.to_csv(all_filename, sep="\t")
    X_train.to_csv(train_filename, sep="\t")
    X_test.to_csv(test_filename, sep="\t")


def survival_histograms(y, hist_bins: int = 10, cum_hist_bins: int = 15):
    y_range = (0.0, 1.25 * max(y))
    # Histogram of patients.
    p, edges = np.histogram(y, range=y_range, bins=hist_bins)
    t = (edges[:-1] + edges[1:]) / 2.0

    # Cumulative histogram.
    res = sp.stats.cumfreq(y, numbins=cum_hist_bins, defaultreallimits=(0.0, max(y)))
    t_cum = res.lowerlimit + np.linspace(
        0, res.binsize * res.cumcount.size, res.cumcount.size
    )
    n_survive = len(y) - res.cumcount

    # Remove last element which contains 0 survivors.
    t_cum = t_cum[:-1]
    n_survive = n_survive[:-1]

    return ((t, p), (t_cum, n_survive))


def merge_mutation_spreadsheet_t0_with_t1(spread_sheet):
    """
    The mutation spreadsheet contains rows for t0 and for t1. Merge these rows.
    """
    # Determine whether it is t0 or t1 measurement.
    spread_sheet["Sample ID"] = spread_sheet["Sample ID"].apply(
        lambda x: int(x.split("_")[1])
    )
    # Replace NA with empty string in order to join the rows using pandas.
    spread_sheet["Coding Change"] = (
        spread_sheet["Coding Change"].astype("string").fillna("")
    )

    # This triplet uniquely defines a record.
    join_columns = ["Patient ID", "Gene", "Coding Change"]

    # Split the spreadsheet in two: One for time point t0, and one for t1.
    t0_samples = spread_sheet["Sample ID"] == 0
    t1_samples = spread_sheet["Sample ID"] == 1
    # Ignore all samples that are not 0 or 1.
    data_frame_t0 = spread_sheet[t0_samples]
    data_frame_t1 = spread_sheet[t1_samples]

    # Verify the assumption that the `join_columns` triplet is unique.
    assert len(data_frame_t0[data_frame_t0[join_columns].duplicated()]) == 0
    assert len(data_frame_t1[data_frame_t1[join_columns].duplicated()]) == 0

    # Change the column names according to timepoint.
    t0_new_names = {
        "Allele Fraction": "T0: Allele Fraction",
        "No. Mutant Molecules per mL": "T0: Mutant concentration",
        "CNV Score": "T0: CNV Score",
    }
    t1_new_names = {
        "Allele Fraction": "T1: Allele Fraction",
        "No. Mutant Molecules per mL": "T1: Mutant concentration",
        "CNV Score": "T1: CNV Score",
    }
    data_frame_t0 = data_frame_t0.rename(columns=t0_new_names).copy()
    data_frame_t1 = data_frame_t1.rename(columns=t1_new_names)

    # Ignore all but the following three columns.
    columns_to_add = [
        "T1: Allele Fraction",
        "T1: Mutant concentration",
        "T1: CNV Score",
    ]
    data_frame_t1 = data_frame_t1[join_columns + columns_to_add]

    # Merge the two data frames back together.
    merged_data_frame = data_frame_t0.set_index(
        ["Patient ID", "Gene", "Coding Change"]
    ).join(
        data_frame_t1.set_index(["Patient ID", "Gene", "Coding Change"]), how="outer"
    )

    # Keep only the columns we are interested in.
    merged_column_names = list(sorted(t0_new_names.values())) + list(
        sorted(t1_new_names.values())
    )
    merged_data_frame = merged_data_frame[merged_column_names]
    return merged_data_frame.copy()
