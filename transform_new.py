import numpy as np
import pandas as pd

from const import clinical_features, meta_columns


def transform_genomic(data: pd.DataFrame, granularity: str) -> dict:
    """
    Apply transformations to the points of the mutations.
    """
    column_pair = sorted(data.columns)
    assert len(column_pair) == 2
    t0_sheet = data[column_pair[0]].copy()
    t1_sheet = data[column_pair[1]].copy()

    def up_or_down(t0, t1):
        return np.sign(t1 - t0)

    transf_sheet = up_or_down(t0_sheet, t1_sheet).copy()

    sheets = {
        "t0": t0_sheet,
        "t1": t1_sheet,
        "up_or_down": transf_sheet.astype(int),
    }

    processed_sheets = {}
    for time_name, mutation_series in sheets.items():
        # Variants with zero value were not called on this time point, so filter.
        observed_mutations = mutation_series[mutation_series != 0.0]
        observed_groups = observed_mutations.groupby(["Patient ID", granularity])
        # Aggregate individual variants.
        processed_sheets[time_name] = _transpose(observed_groups.sum())
        processed_sheets[time_name + "_TMB"] = _transpose(
            observed_groups.count()
        ).astype(int)

    processed_sheets["t0_indicator"] = _as_indicators(t0_sheet, granularity=granularity)
    processed_sheets["t1_indicator"] = _as_indicators(t1_sheet, granularity=granularity)
    return processed_sheets


def transform_clinic(data: pd.DataFrame) -> pd.DataFrame:
    """
    Further curation of the clinical data.
    """
    X = data.copy()

    # After inspection the other histologies are partially adeno.
    X["histology_grouped"].replace({"other": "adeno"}, inplace=True)
    first_line_therapy = X["therapyline"].isin([0, 1])

    # According to the selection criterea of the study, 0th and 1st line should be
    # combined.
    X.loc[first_line_therapy, "therapyline"] = "0+1"
    X.loc[~first_line_therapy, "therapyline"] = ">1"

    # Unknown smoking status are probably smokers according Harry.
    X["smokingstatus"].replace({"unknown": "smoker"}, inplace=True)
    # Group together current and previous smokers.
    X["smokingstatus"].replace(
        {"previous": "current+previous", "smoker": "current+previous"}, inplace=True
    )

    d = {"no metastasis present": "absent", "metastasis present": "present"}
    X[meta_columns] = X[meta_columns].replace(d)

    # Partition age in two.
    young = X["Age"] < 65
    X["age"] = r"$\geq$ 65"
    X.loc[young, "age"] = "<65"
    # Remove original column.
    X.drop(columns="Age", inplace=True)

    X["PD_L1>50%"] = np.nan
    not_null = X["PD_L1_continous"].notnull()
    X.loc[not_null, "PD_L1>50%"] = (X.loc[not_null, "PD_L1_continous"] > 50).astype(int)
    X.drop(columns="PD_L1_continous", inplace=True)

    X["clearance"] = X["T0_T1"].map(
        {
            "only tumor derived mutation at t0": "t1",
            "only tumor derived mutation at t1": "t0",
            "patient had tumor derived mutations at both timepoints": "no",
            "no tumor derived mutations at t0 and t1": "t0+t1",
        }
    )
    X.drop(columns="T0_T1", inplace=True)

    new_clinical_features = clinical_features.copy()
    new_clinical_features[new_clinical_features.index("T0_T1")] = "clearance"
    new_clinical_features[new_clinical_features.index("Age")] = "age"
    new_clinical_features[new_clinical_features.index("PD_L1_continous")] = "PD_L1>50%"
    # All clinical variables are categories.
    X[new_clinical_features] = X[new_clinical_features].astype("category")
    return X


def _transpose(series: pd.Series, columns=None) -> pd.DataFrame:
    """
    Transpose level 1 values (gene, or pathway/network) of series into table.
    """
    # Convert to Series if it a 1D dataframe.
    if not isinstance(series, pd.Series):
        print("type", type(series))
        print("shape", series.shape)
        assert series.shape[1] == 1
        series = series.iloc[:, 0]

    patient_ids = series.index.get_level_values("Patient ID")
    if columns is None:
        columns = sorted(np.unique(series.index.get_level_values(1)))
    transposed_sheet = pd.DataFrame(
        0.0, index=np.unique(patient_ids), columns=sorted(columns)
    )
    for ptid, column in series.index:
        transposed_sheet.loc[ptid, column] = series.loc[ptid, column]

    return transposed_sheet


def _as_indicators(
    series: pd.Series, granularity: str, suffix: str = ""
) -> pd.DataFrame:
    """
    Make indicator variables marking the presence of a mutation.
    """
    coarseness = ["Patient ID", granularity]
    presence = series.groupby(coarseness).sum()
    presence[:] = 1
    presence = _transpose(presence)
    new_column_names = [f"{c}{suffix}" for c in presence.columns]
    presence.columns = new_column_names
    return presence.copy()

