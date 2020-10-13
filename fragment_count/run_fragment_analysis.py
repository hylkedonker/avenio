import datetime
import glob
import logging
from pathlib import Path
import sys

import pandas as pd

from analysis import compute_variant_fragment_statistics
from utils import select_sample_variants

# Specify and create output directories.
output_folder = Path("/package/output2/")
output_pbmc = output_folder / "pbmc_plus_plasma"
output_pbmc.mkdir(parents=True, exist_ok=True)
output_tumor = output_folder / "tumor_derived"
output_tumor.mkdir(parents=True, exist_ok=True)
output_chip = output_folder / "chip"
output_chip.mkdir(parents=True, exist_ok=True)
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
        compute_variant_fragment_statistics(
            sample_folder, output_pbmc, variant_metadata=pbmc_metadata
        )
    else:
        tumor_metadata, chip_metadata = select_sample_variants(sample_folder.name)
        compute_variant_fragment_statistics(
            sample_folder, output_tumor, variant_metadata=tumor_metadata
        )
        compute_variant_fragment_statistics(
            sample_folder, output_chip, variant_metadata=chip_metadata
        )

    # run_metadata
logging.info("Done!")
