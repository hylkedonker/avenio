import numpy as np
import pandas as pd

from const import clinical_features, outcome_labels
from extract import extract_avenio_mutations, extract_clinical_sheet
from transform import transform_clinic, transform_genomic


def build_clinical_genomic_sheet(granularity="chromosome") -> pd.DataFrame:
    """
    Extract variants (t0+t1) and clinical information and merge into table.
    """
    variants_per_column = extract_avenio_mutations()

    # Build SNVs/InDels columns.
    concentration_sheets = transform_genomic(
        variants_per_column["No. Mutant Molecules per mL"], granularity=granularity
    )
    # Build copy number alteration columns (including presence indicator
    # columns).
    cna_sheets = transform_genomic(
        variants_per_column["CNV Score"], granularity=granularity
    )
    cna = cna_sheets["up_or_down"].join(
        cna_sheets["t0_indicator"], how="outer", lsuffix="_cna", rsuffix="_cna_at_t0"
    )

    X_genomic = (
        concentration_sheets["up_or_down"].join(cna, how="outer").fillna(0).astype(int)
    )

    # Avenio targeted panel captures 77 genes, 192 KB.
    exome_captured = 0.192
    # Sum over all genes/chromosomes/pathways.
    normalized_tmb_t0 = (
        concentration_sheets["t0_tmb"].sum(axis=1)
        / concentration_sheets["t0"].sum(axis=1)
        / exome_captured
    )
    normalized_tmb_t1 = (
        concentration_sheets["t1_tmb"].sum(axis=1)
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
    return _encode_as_numeric(X.dropna(subset=["response_grouped"]))


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


def _encode_as_numeric(X):
    """
    Turn categorical covariates into dummies.
    """
    X["therapy"] = (
        X.pop("Systemischetherapie")
        .str.lower()
        .replace(
            {
                "nivolumab + ipilimumab": "nivolumab+ipilimumab",
                "ipi-novu": "nivolumab+ipilimumab",
            }
        )
    )
    clearance_dummies = pd.get_dummies(X[["clearance"]])
    # Drop first column (no clearance).
    clearance_columns = ["clearance_t0", "clearance_t0+t1", "clearance_t1"]

    tmb_features = ["normalized_tmb_t0", "normalized_tmb_t1"]
    black_list = tmb_features + ["pd_l1>50%", "clearance"]
    X_prime = X.drop(columns=black_list)

    categories_to_encode = [
        "age",
        "gender",
        "therapyline",
        "smokingstatus",
        "histology",
        "stage",
        "therapy",
        "ECOG_PS",
        "lymfmeta",
        "brainmeta",
        "adrenalmeta",
        "livermeta",
        "lungmeta",
        "skeletonmeta",
    ]

    def to_lower(column):
        """ Make text columns lowercase. """
        if hasattr(column, "str"):
            return column.str.lower()
        return column

    X_prime = pd.get_dummies(X[categories_to_encode].apply(to_lower), drop_first=True)

    genetic_columns = (
        set(X.columns)
        - set(categories_to_encode)
        - set(black_list)
        - set(outcome_labels)
    )
    genetic_columns = sorted(genetic_columns)

    genetic_direction = sorted(x for x in genetic_columns if "cna_at_t" not in x)
    genetic_presence = sorted(set(genetic_columns) - set(genetic_direction))
    gene_up = [x + "↑" for x in genetic_direction]
    gene_down = [x + "↓" for x in genetic_direction]
    X_prime[gene_up] = X[genetic_direction].applymap(lambda x: 1 if x > 0 else 0)
    X_prime[gene_down] = X[genetic_direction].applymap(lambda x: 1 if x < 0 else 0)
    X_prime[genetic_presence] = X[genetic_presence]

    X_prime[clearance_columns] = clearance_dummies[clearance_columns]
    X_prime["pd_l1>50%"] = X["pd_l1>50%"]
    X_prime[tmb_features] = X[tmb_features]

    # Curate clinical outcomes.
    X_prime["clinical_response"] = X["Clinical_Response"].str.lower()
    X_prime["response_grouped"] = (
        X["response_grouped"]
        .str.lower()
        .map(
            {
                "non responder (sd+pd)": "non responder (sd+pd+ne)",
                "non evaluable (ne)": "non responder (sd+pd+ne)",
                "responder (pr+cr)": "responder (pr+cr)",
            }
        )
    )
    X_prime[["os_months", "pfs_months"]] = X[["OS_months", "PFS_months"]]
    X_prime["pfs_event"] = (
        X["Censor_progression"]
        .str.lower()
        .map({"progression of disease": 1, "no progression of disease": 0})
    )
    X_prime["os_event"] = X["Censor_OS"]
    X_prime["pfs>1yr"] = (X["PFS_months"] > 12).astype(int)

    return X_prime.copy()
