from typing import Callable, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from const import clinical_features
from source import (
    add_mutationless_patients,
    load_avenio_files,
    phenotype_labels,
    read_preprocessed_data,
)
from pipelines import pipeline_Freeman, pipeline_Richard


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


def transform_column_pair(
    data_frame: pd.DataFrame,
    transformation: Callable = lambda x, y: y - x,
    column_pair: List[str] = ["T0: Allele \nFraction", "T1: Allele Fraction"],
    handle_duplicates="sum",
) -> pd.DataFrame:
    """
    For each patient, pair-wise transform the values in `column_pair`.

    For each mutation for a given patient, calculate `transformation(f_t0,
    f_t1)` from the allele frequencies measured as t0 and t1.
    """
    gene_vocabulary = sorted(data_frame["Gene"].unique())

    allowed_duplicate_actions = ("min", "max", "ignore", "concat", "sum")
    if handle_duplicates.lower() not in allowed_duplicate_actions:
        raise ValueError(
            "Allowed values for handle_duplicates are {}.".format(
                allowed_duplicate_actions
            )
        )

    # The transformation is calculated between two columns (t0 and t1).
    if len(column_pair) != 2:
        raise ValueError("Allele frequency columns must be precisely two!")

    # The columns that were passed must actually exist.
    column_names = data_frame.columns
    if column_pair[0] not in column_names or column_pair[1] not in column_names:
        raise KeyError(
            "Column lookup error in `allel_freq_columns` = {}.".format(column_pair)
        )

    # There may not be any NA values in the columns.
    if (
        sum(data_frame[column_pair[0]].isna()) > 0
        or sum(data_frame[column_pair[1]].isna()) > 0
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
        f_t0, f_t1 = (grouped[column_pair[0]], grouped[column_pair[1]])

        # Aggregate results _after_ transformation.
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
    input_data: pd.DataFrame, columns_to_number: list, fill_ND=None
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

    # 1) Replace "ND" and "Absent" values with NA.
    na_map = {"ND": fill_ND, "Absent": fill_ND}
    clean_data[columns_to_number] = clean_data.loc[:, columns_to_number].applymap(
        lambda x: na_map[x] if x in na_map else x
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


def coarse_grain_columns(
    data_frame: pd.DataFrame, operation: Callable = np.sum
) -> pd.DataFrame:
    """
    Coarse grain over coding changes.

    Columns have the form, e.g., ('hsa05226', 'ABL1', 'c.2976G>A'), coarse grain over
    the last element (the coding change in this case).

    Args:
        drop_remaining_grain_labels (bool): Whether to drop 'hsa05226' in the example
            above.
    """
    grain_labels = tuple(zip(*data_frame.columns))
    coarse_labels = tuple(zip(*grain_labels[:-1]))  # Repack all but the last element.

    unique_labels = np.unique(coarse_labels, axis=0)
    coarse_sheet = pd.DataFrame(0.0, index=data_frame.index, columns=unique_labels)

    # Select columns belonging to `gene` and aggregate using `operation`.
    for label in coarse_sheet.columns:

        def select_grains(x):
            """ Select all columns belonging to `label` coarseness. """
            return x[:-1] == label

        grains = list(filter(select_grains, data_frame.columns))
        coarse_sheet[label] = data_frame[grains].aggregate(operation, axis="columns")

    return coarse_sheet


def transpose_and_transform(
    data_frame: pd.DataFrame,
    column_pair: list,
    transformation: Callable[[float, float], float],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Tranpose `data_frame`, returning the t0 (column_pair[0]), t1 (column_pair[1]) and
    transformation(t0, t1) values.

    Transforms `data_frame` with columns ('Patient ID', 'Gene', 'Coding Change') to a
    data frame with layout 'Patient ID' x ('Gene', 'Coding Change', 'Pathway').
    """
    patient_ids = data_frame["Patient ID"]
    # Assign pathway to each gene.
    pathways = pd.read_excel("gene_pathway_most_frequent.xlsx")
    gene_path_map = dict(zip(pathways.iloc[:, 0], pathways.iloc[:, 1]))
    data_frame["Pathway"] = data_frame["Gene"].map(gene_path_map)

    # New column name is a pair defined by the values of these two fields.
    names = ["Pathway", "Gene", "Coding Change"]
    new_columns = tuple(zip(*(data_frame[c] for c in names)))

    # Create and fill three transposed sheets: t0, t1 and f(t0, t1).
    t0_sheet = pd.DataFrame(
        0.0, index=np.unique(patient_ids), columns=np.unique(new_columns, axis=0)
    )
    t1_sheet = t0_sheet.copy()
    transf_sheet = t0_sheet.copy()
    for ind in data_frame.index:
        patient_id = data_frame.loc[ind, "Patient ID"]
        column_name = tuple(data_frame.loc[ind, c] for c in names)
        new_ind = (patient_id, column_name)
        t0_value = data_frame.loc[ind, column_pair[0]]
        t1_value = data_frame.loc[ind, column_pair[1]]
        t0_sheet.loc[new_ind] = t0_value
        t1_sheet.loc[new_ind] = t1_value
        transf_sheet.loc[new_ind] = transformation(t0_value, t1_value)

    return t0_sheet.copy(), t1_sheet.copy(), transf_sheet.copy()


def load_process_and_store_spreadsheets(
    spread_sheet_filename: str = "variant_list_20200409.xlsx",
    spss_filename: str = "clinical_20200420.sav",
    transformation: Callable = lambda x, y: y - x,
    columns: List[str] = [
        "Allele Fraction",
        "No. Mutant Molecules per mL",
        "CNV Score",
    ],
    all_filename_prefix: str = "output/all",
    train_filename_prefix: str = "output/train",
    test_filename_prefix: str = "output/test",
):
    """
    Read, clean, transform, and store raw data.

    1) Load the mutation Excel spreadsheet and the phenotype SPSS file.
    2) Transform and transpose the mutation columns.
    3) Combine mutation and phenotype data.
    4) Split and store data to disk.
    """
    # Load data from spreadsheet and SPSS files.
    patient_mutations, clinical_data = load_avenio_files(
        spread_sheet_filename, spss_filename
    )

    # Combine the T0 and T1 measurements in a single record.
    spread_sheet = merge_mutation_spreadsheet_t0_with_t1(patient_mutations, columns)

    # Make a different document for each of the column pairs.
    for column in columns:
        column_pair = [f"T0: {column}", f"T1: {column}"]
        # Filter out the column pairs, and remove empty fields.
        mutations = spread_sheet[column_pair].dropna(how="all").fillna(0).reset_index()
        # Repair dirty cells (with per cent signs etc.).
        clean_mutation_sheet = clean_and_verify_data_frame(mutations, column_pair)

        # Transpose table, and generate sheet for t0 mutations, t1 mutations, and those
        # combined with `transformation`.
        transposed_sheets = transpose_and_transform(
            clean_mutation_sheet, column_pair, transformation
        )
        timepoint_names = ("t0", "t1", transformation.__name__)
        for time_name, mutation_sheet in zip(timepoint_names, transposed_sheets):
            gene_sheet = coarse_grain_columns(mutation_sheet, operation=np.sum)
            pathway_sheet = coarse_grain_columns(gene_sheet, operation=np.sum)
            gene_sheet.columns = gene_sheet.columns.to_series().apply(lambda x: x[1])
            pathway_sheet.columns = pathway_sheet.columns.to_series().apply(
                lambda x: x[0]
            )

            # Make a sheet for each level of coarseness.
            for coarseness_name, coarse_sheet in {
                "gene": gene_sheet,
                "pathway": pathway_sheet,
            }.items():
                coarse_sheet = add_mutationless_patients(coarse_sheet, clinical_data)
                # Merge with clinical data.
                final_spreadsheet = merge_mutations_with_phenotype_data(
                    coarse_sheet, clinical_data
                )

                # Check that all patients are included.
                assert final_spreadsheet.shape[0] == clinical_data.shape[0]

                # And store to disk.
                data_frame_to_disk(
                    final_spreadsheet,
                    all_filename=all_filename_prefix
                    + f"_{coarseness_name}__{time_name}__{column}",
                    train_filename=train_filename_prefix
                    + f"_{coarseness_name}__{time_name}__{column}",
                    test_filename=test_filename_prefix
                    + f"_{coarseness_name}__{time_name}__{column}",
                )


def data_frame_to_disk(
    X: pd.DataFrame,
    all_filename: str,
    train_filename: str,
    test_filename: str,
    random_state: int = 1234,
):
    """
    Write data to disk.
    """
    f_test = 0.2
    X.sort_index(inplace=True)
    X_train, X_test = train_test_split(X, test_size=f_test, random_state=random_state)
    X.to_csv(all_filename + ".tsv", sep="\t")
    X.to_excel(all_filename + ".xlsx")
    # X.to_sav(all_filename + ".sav")
    X_train.to_csv(train_filename + ".tsv", sep="\t")
    X_test.to_csv(test_filename + ".tsv", sep="\t")


def clean_and_verify_data_frame(
    mutation_data_frame: pd.DataFrame, columns_to_transform: list
) -> pd.DataFrame:
    """
    Clean up data frame records, verify consistency, and transpose to gene columns.
    """
    # Convert columns to numbers and drop rows with missing data.
    clean_patient_mutations, dirty_patient_mutations = clean_mutation_columns(
        mutation_data_frame, columns_to_number=columns_to_transform
    )

    clean_patients = clean_patient_mutations["Patient ID"].unique()
    dirty_patients = dirty_patient_mutations["Patient ID"].unique()

    # Verify that the combination of the patients must give all patients.
    assert set(clean_patients).union(set(dirty_patients)) == set(
        mutation_data_frame["Patient ID"].unique()
    )

    # Verify that there are no more NA values.
    assert (
        clean_patient_mutations[columns_to_transform].dropna().shape[0]
        == clean_patient_mutations[columns_to_transform].shape[0]
    )
    # And that everything went in to the "dirty" records.
    assert dirty_patient_mutations[columns_to_transform].dropna().shape[0] == 0

    return clean_patient_mutations


def merge_mutations_with_phenotype_data(
    transposed_mutation_data_frame: pd.DataFrame, phenotype_data_frame: pd.DataFrame
) -> pd.DataFrame:
    phenotypes_to_keep = clinical_features + phenotype_labels
    # Combine mutation data and phenotype data.
    X = pd.merge(
        left=transposed_mutation_data_frame,
        right=phenotype_data_frame[phenotypes_to_keep],
        left_index=True,
        right_index=True,
    )
    X.dropna(subset=["response_grouped"], inplace=True)
    return X


def merge_mutation_spreadsheet_t0_with_t1(
    spread_sheet: pd.DataFrame, columns: List[str]
) -> pd.DataFrame:
    """
    The mutation spreadsheet contains rows for t0 and for t1. Merge these rows.
    """

    def get_sample_number(sample_id: str):
        tokens = str(sample_id).split("_")
        if len(tokens) > 1:
            return int(tokens[1])
        return None

    # Determine whether it is t0 or t1 measurement.
    spread_sheet["Sample ID"] = spread_sheet["Sample ID"].apply(get_sample_number)
    # Replace NA with empty string in order to join the rows using pandas.
    spread_sheet["Coding Change"] = spread_sheet["Coding Change"].astype(str).fillna("")

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
    t0_new_names = {column: f"T0: {column}" for column in columns}
    t1_new_names = {column: f"T1: {column}" for column in columns}
    data_frame_t0 = data_frame_t0.rename(columns=t0_new_names).copy()
    data_frame_t1 = data_frame_t1.rename(columns=t1_new_names)

    # Ignore all but the columns that uniquely identify a record, plus the columns of
    # interest.
    t1_columns_to_keep = join_columns + list(sorted(t1_new_names.values()))
    data_frame_t1 = data_frame_t1[t1_columns_to_keep]

    # Merge the two data frames back together.
    merged_data_frame = data_frame_t0.set_index(
        ["Patient ID", "Gene", "Coding Change"]
    ).join(
        data_frame_t1.set_index(["Patient ID", "Gene", "Coding Change"]), how="outer"
    )

    # Keep only the columns we are interested in (i.e., remove the remaining columns
    # from t0).
    merged_column_names = list(sorted(t0_new_names.values())) + list(
        sorted(t1_new_names.values())
    )
    merged_data_frame = merged_data_frame[merged_column_names]
    return merged_data_frame.copy()


def combine_tsv_files(
    sheet_a: str, sheet_b: str, suffixes=("_snv", "_cnv")
) -> pd.DataFrame:
    """
    Merge two TSV files with the same Patient ID indices.
    """
    X_train_a, y_train_a = read_preprocessed_data(sheet_a)
    X_train_b, y_train_b = read_preprocessed_data(sheet_b)

    # Check consistency of data frames and combine.
    assert set(X_train_a.index.unique()) == set(X_train_b.index.unique())

    # Remove phenotype columns to get the genetic columns.
    genetic_columns_b = list(set(X_train_b.columns) - set(clinical_features))
    genetic_columns_b.sort()
    # We want to rename only the genetic columns, not the phenotypes.
    genetic_columns_a = list(set(X_train_a.columns) - set(clinical_features))
    rename_table_a = {
        column_name: column_name + suffixes[0] for column_name in genetic_columns_a
    }

    # Merge the feature and the label frames.
    X_train = X_train_a.rename(columns=rename_table_a).merge(
        X_train_b[genetic_columns_b].rename(columns=lambda x: x + suffixes[1]),
        left_index=True,
        right_index=True,
        how="outer",
        suffixes=suffixes,
    )
    columns_to_drop = [c for c in X_train.columns if "_y" in c]
    X_train.drop(columns=columns_to_drop, inplace=True)
    assert y_train_a.equals(y_train_b)
    y_train = y_train_a.copy()

    # Validate consistency of features with labels.
    assert set(X_train.index.unique()) == set(y_train.index.unique())

    return X_train, y_train


def generate_data_pairs(filename_prefix: str, snv_type: str) -> dict:
    """
    Combine SNV and CNV data for t0, t1, dt and harmonic mean in (X, y) pairs.
    """
    datasets = [
        "difference",
        "harmonic_mean",
        "relative_difference",
        "up_or_down",
        "t0",
        "t1",
    ]
    pos_label = "responder (pr+cr)"
    named_pairs = {}
    for data_type in datasets:
        type_prefix = f"{filename_prefix}__{data_type}"
        X, y = combine_tsv_files(
            f"{type_prefix}__{snv_type}.tsv", f"{type_prefix}__CNV Score.tsv"
        )
        y = y["response_grouped"] == pos_label
        named_pairs[data_type] = (X, y)
    return named_pairs


def generate_model_data_pairs(data_pairs: dict, model_parameters: dict) -> dict:
    """
    Combine data sets for different time points in (model, (X, y)) pairs.

    See `generate_data_pairs`.
    """
    logistic_Freeman = pipeline_Freeman(LogisticRegression(**model_parameters))
    logistic_Richard = pipeline_Richard(LogisticRegression(**model_parameters))
    return {
        "Clinical": (logistic_Richard, data_pairs["difference"]),
        "Clinical +\n Genomic $t_0$": (logistic_Freeman, data_pairs["t0"]),
        "Clinical +\n Genomic $t_1$": (logistic_Freeman, data_pairs["t1"]),
        "Clinical +\n Genomic $\Delta t$": (logistic_Freeman, data_pairs["difference"]),
        "Clinical +\n Genomic $f$": (
            logistic_Freeman,
            data_pairs["relative_difference"],
        ),
        "Clinical +\n Genomic $h$": (logistic_Freeman, data_pairs["harmonic_mean"]),
        "Clinical +\n Genomic $\iota$": (logistic_Freeman, data_pairs["up_or_down"]),
    }
