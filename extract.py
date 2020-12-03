from typing import Dict, List, Tuple

import pandas as pd

from const import clinical_features, outcome_labels


def extract_avenio_mutations(
    columns=["Allele Fraction", "No. Mutant Molecules per mL", "CNV Score"]
) -> Dict[str, pd.DataFrame]:
    """
    Extract mutations from spreadsheet per column.
    """
    # Load data from spreadsheet.
    patient_mutations = pd.read_excel("variant_list_20200730.xlsx", sheet_name=1)

    # Add pathway or network annotations.
    patient_mutations = _annotate_genes(patient_mutations, "gene_annotation.xlsx")

    # Combine the T0 and T1 measurements in a single record.
    patient_mutations["chromosome"] = patient_mutations["Genomic Position"].apply(
        lambda x: x.split(":")[0] if isinstance(x, str) else x
    )
    # Make seperate timepoint columns for the following columns in the
    # spreadsheet.

    spread_sheet = _merge_mutation_spreadsheet_t0_with_t1(patient_mutations, columns)

    cleaned_sheets = {}
    # Make a different document for each of the column pairs.
    for column in columns:
        column_pair = [f"T0: {column}", f"T1: {column}"]
        # Filter out the column pairs, and remove empty fields.
        mutations = spread_sheet[column_pair].dropna(how="all").fillna(0)
        # Repair dirty cells (with per cent signs etc.).
        clean_mutation_sheet = _clean_and_verify_avenio_sheet(mutations, column_pair)
        cleaned_sheets[column] = clean_mutation_sheet

    return cleaned_sheets


def extract_clinical_sheet() -> pd.DataFrame:
    """
    Load the clinical information from SPSS file.
    """
    clinical_sheet = pd.read_spss("clinical_20200420.sav")

    # And set Patient ID as index.
    clinical_sheet["studynumber"].name = "Patient ID"
    clinical_sheet.set_index(clinical_sheet["studynumber"].astype(int), inplace=True)
    # Convert stage from float to integer.
    columns_to_int = ["stage", "therapyline"]
    clinical_sheet[columns_to_int] = clinical_sheet[columns_to_int].astype(int)

    return clinical_sheet[clinical_features + outcome_labels]


def _annotate_genes(data_frame: pd.DataFrame, annotation_filename: str):
    """
    Use spreadsheet with gene annotations (e.g., pathway or network).
    """
    annotated = data_frame.copy()
    # Assign pathway to each gene.
    pathways = pd.read_excel(annotation_filename)
    gene_path_map = dict(zip(pathways.iloc[:, 0], pathways.iloc[:, 1]))
    annotated["Annotation"] = annotated["Gene"].map(gene_path_map)
    return annotated


def _merge_mutation_spreadsheet_t0_with_t1(
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

    spread_sheet = spread_sheet.copy()
    # Determine whether it is t0 or t1 measurement.
    spread_sheet["Sample ID"] = spread_sheet["Sample ID"].apply(get_sample_number)
    # Replace NA with empty string in order to join the rows using pandas.
    spread_sheet["Coding Change"] = spread_sheet["Coding Change"].astype(str).fillna("")

    # This triplet uniquely defines a record.
    join_columns = [
        "Patient ID",
        "chromosome",
        "Gene",
        "Annotation",
        "Variant Description",
        "Coding Change",
    ]

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
    merged_data_frame = data_frame_t0.set_index(join_columns).join(
        data_frame_t1.set_index(join_columns), how="outer"
    )

    # Keep only the columns we are interested in (i.e., remove the remaining columns
    # from t0).
    merged_column_names = list(sorted(t0_new_names.values())) + list(
        sorted(t1_new_names.values())
    )
    merged_data_frame = merged_data_frame[merged_column_names]
    return merged_data_frame.copy()


def _clean_and_verify_avenio_sheet(
    mutation_data_frame: pd.DataFrame, columns_to_transform: list
) -> pd.DataFrame:
    """
    Clean up data frame records, verify consistency, and transpose to gene columns.
    """
    # Convert columns to numbers and drop rows with missing data.
    clean_patient_mutations, dirty_patient_mutations = _clean_cells(
        mutation_data_frame, columns_to_number=columns_to_transform
    )

    clean_patients = clean_patient_mutations.index.get_level_values(
        "Patient ID"
    ).unique()
    dirty_patients = dirty_patient_mutations.index.get_level_values(
        "Patient ID"
    ).unique()

    # Verify that the combination of the patients must give all patients.
    assert set(clean_patients).union(set(dirty_patients)) == set(
        mutation_data_frame.index.get_level_values("Patient ID").unique()
    )

    # Verify that there are no more NA values.
    assert (
        clean_patient_mutations[columns_to_transform].dropna().shape[0]
        == clean_patient_mutations[columns_to_transform].shape[0]
    )
    # And that everything went in to the "dirty" records.
    assert dirty_patient_mutations[columns_to_transform].dropna().shape[0] == 0

    return clean_patient_mutations


def _clean_cells(
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
