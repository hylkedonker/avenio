from typing import Tuple

import numpy as np
import pandas as pd

RANDOM_STATE = 1234
np.random.seed(RANDOM_STATE)


def load_avenio_files(
    spread_sheet_filename: str = "2019-08-27_PLASMA_DEFAULT_Results_Groningen.xlsx",
    spss_filename: str = "phenotypes.sav",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the mutation spreadsheet and SPSS phenotype data in two data frames.
    """
    # Load data from spreadsheet.
    mutation_data_frame = pd.read_excel(spread_sheet_filename, sheet_name=2)

    # Load the phenotypes from SPSS file.
    phenotypes = pd.read_spss(spss_filename)
    # Remove patients for which there is no sequencing data.
    phenotypes = phenotypes[~phenotypes["VAR00001"].isna()]
    # And set Patient ID as index.
    phenotypes["VAR00001"].name = "Patient ID"
    phenotypes.set_index(phenotypes["VAR00001"].astype(int), inplace=True)

    return (mutation_data_frame, phenotypes)
