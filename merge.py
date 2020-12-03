import numpy as np
import pandas as pd

from const import clinical_features, outcome_labels
from extract import extract_avenio_mutations, extract_clinical_sheet
from transform_new import transform_clinic, transform_genomic


def build_clinical_genomic_sheet() -> pd.DataFrame:
    """
    Extract variants (t0+t1) and clinical information and merge into table.
    """
    variants_per_column = extract_avenio_mutations()

    # Build SNVs/InDels columns.
    concentration_sheets = transform_genomic(
        variants_per_column["No. Mutant Molecules per mL"], granularity="chromosome"
    )
    # Build copy number alteration columns (including presence indicator
    # columns).
    cna_sheets = transform_genomic(variants_per_column["CNV Score"], granularity="Gene")
    cna = cna_sheets["up_or_down"].join(
        cna_sheets["t0_indicator"], how="outer", lsuffix="_CNA", rsuffix="_CNA_at_t0"
    )

    X_genomic = (
        concentration_sheets["up_or_down"].join(cna, how="outer").fillna(0).astype(int)
    )

    # Avenio targeted panel captures 77 genes, 192 KB.
    exome_captured = 0.192
    # Sum over all genes/chromosomes/pathways.
    normalized_tmb_t0 = (
        concentration_sheets["t0_TMB"].sum(axis=1)
        / concentration_sheets["t0"].sum(axis=1)
        / exome_captured
    )
    normalized_tmb_t1 = (
        concentration_sheets["t1_TMB"].sum(axis=1)
        / concentration_sheets["t1"].sum(axis=1)
        / exome_captured
    )

    X_genomic["normalized_tmb_t0"] = normalized_tmb_t0
    X_genomic["normalized_tmb_t1"] = normalized_tmb_t1
    # Those with no mutation have TMB=0.
    X_genomic = X_genomic.fillna(0.0)

    X_clinic = extract_clinical_sheet()
    X_clinic = transform_clinic(X_clinic)

    X_genomic = _add_clearance_patients(X_genomic, X_clinic)

    # Merge with clinical data.
    X = pd.merge(left=X_genomic, right=X_clinic, left_index=True, right_index=True)
    return X.dropna(subset=["response_grouped"])


def _add_clearance_patients(
    mutation_table: pd.DataFrame, clinical_sheet: pd.DataFrame
) -> pd.DataFrame:
    """
    Add mutationless patients to the mutation table by filling the rows with zeros.
    """
    # Add the patients that are not in the mutation list.
    mutationless_patients = set(clinical_sheet.index) - set(mutation_table.index)
    # Make the difference set ordered using `tuple`, and then convert to a Series.
    mutationless_patients = pd.Series(tuple(mutationless_patients), dtype=int)

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
