from typing import Tuple

import pandas as pd

from merge import build_clinical_genomic_sheet
from const import outcome_labels


def load_dataset(granularity="chromosome") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load dataset and split covariates from outcomes.
    """
    outcomes = [
        "clinical_response",
        "response_grouped",
        "os_months",
        "pfs_months",
        "pfs>1yr",
        "pfs_event",
        "os_event",
    ]
    X = build_clinical_genomic_sheet(granularity)
    return X.drop(columns=outcomes).copy(), X[outcomes].copy()
