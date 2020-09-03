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
    run_metadata = Path("/metadata/variant_list_20200730.xlsx")
    patient_id, sample_type = sample_folder_name.split("_")
    patient_id = int(patient_id)

    sheets = pd.read_excel(run_metadata, sheet_name=[1, 2, 3])
    tumor_sheet = sheets[1]
    pbmc_plasma_sheet = sheets[2]
    chip_sheet = sheets[3]

    columns_to_keep = ["Gene", "Coding Change", "Genomic Position", "Mutation Class"]
    if "PBMC" in sample_folder_name:
        # Keep PBMC variants from both timepoints. We will use `drop_duplicates` below
        # for variants that are both present.
        pbmc_constraint = pbmc_plasma_sheet["Patient ID"] == patient_id
        return pbmc_plasma_sheet[pbmc_constraint].drop_duplicates().copy()

    tumor_constraint = tumor_sheet["Sample ID"] == sample_folder_name
    chip_constraint = chip_sheet["Sample ID"] == sample_folder_name
    # Select the genomic position from the pbmc-plasma sheet, because it is missing in
    # the chip sheet.
    tumors = tumor_sheet[tumor_constraint]
    chips = chip_sheet[chip_constraint].merge(
        pbmc_plasma_sheet[pbmc_plasma_sheet["Sample ID"] == sample_folder_name],
        on=["Gene", "Coding Change", "Mutation Class"],
        how="inner",
        suffixes=("", "_y"),
    )

    return (
        tumors.filter(items=columns_to_keep).drop_duplicates().copy(),
        chips.filter(items=columns_to_keep).drop_duplicates().copy(),
    )


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
        pbmc_metadata = select_sample_variants(sample_folder.name)
        compute_variant_fragment_size_counts(
            sample_folder,
            output_folder / "pbmc_plus_plasma",
            variant_metadata=pbmc_metadata,
        )
    else:
        tumor_metadata, chip_metadata = select_sample_variants(sample_folder.name)

        compute_variant_fragment_size_counts(
            sample_folder,
            output_folder / "tumor_derived",
            variant_metadata=tumor_metadata,
        )
        compute_variant_fragment_size_counts(
            sample_folder, output_folder / "chip", variant_metadata=chip_metadata
        )

    # run_metadata
logging.info("Done!")
