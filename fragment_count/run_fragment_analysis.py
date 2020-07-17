import datetime
import glob
import logging
from pathlib import Path
import sys

import pandas as pd

from analysis import compute_variant_fragment_size_counts

output_folder = Path("/package/output/")
log_folder = Path("/package/log")

# Set up logging to a file.
timestamp = datetime.datetime.now().strftime(r"%Y%m%d_%H%M")
logfile = log_folder / (Path(sys.argv[1]).name + "__" + timestamp + ".log")
logging.basicConfig(
    filename=logfile,
    level=logging.INFO,
    format=r"%(asctime)s %(levelname)-8s %(message)s",
    datefmt=r"%Y-%m-%d %H:%M:%S",
)


def select_sample_variants(sample_folder_name: str) -> pd.DataFrame:
    """ Find variants for this run from the spreadsheet run file. """
    run_metadata = Path("/metadata/variant_list_20200409.xlsx")
    patient_id, sample_type = sample_folder_name.split("_")
    patient_id = int(patient_id)

    sheet_number = 1  # Tumor-derived variants sheet.
    if sample_type == "PBMC":
        sheet_number = 2  # Variants PBMC & plasma sheet.

    run_sheet = pd.read_excel(run_metadata, sheet_name=sheet_number)

    if "PBMC" in sample_folder_name:
        # Keep both PBMC variants that are eitehr in t0 or in t1. We will use
        # `drop_duplicates` below for variants that are both present.
        constraint = run_sheet["Patient ID"] == patient_id
    else:
        constraint = run_sheet["Sample ID"] == sample_folder_name
    run_variants = run_sheet[constraint]

    columns_to_keep = ["Gene", "Coding Change", "Genomic Position"]
    return run_variants[columns_to_keep].drop_duplicates().copy()


path_pattern = Path(sys.argv[1]) / r"*Expanded*/*"
for sample in glob.glob(str(path_pattern)):
    sample_folder = Path(sample)
    sample_suffix = sample.split("_")[-1]
    if sample_suffix not in ("0", "1", "PBMC"):
        logging.info(
            f"Skipping {sample_folder.name}, because presumably it is neither a t={{0,1}} timepoint nor a PBMC."
        )
        continue
    logging.info(f"Computing fragment counts for {sample_folder.name}")

    if sample_suffix == "PBMC":
        sample_output_location = output_folder / "pbmc_plus_plasma"
    else:
        sample_output_location = output_folder / "tumor_derived"

    run_metadata = select_sample_variants(sample_folder.name)
    compute_variant_fragment_size_counts(
        sample_folder, sample_output_location, variant_metadata=run_metadata
    )
logging.info("Done!")
