import datetime
import glob
import logging
from pathlib import Path
import sys

from analysis import compute_variant_fragment_size_counts

run_metadata = Path("/metadata/variant_list_20200409.xlsx")
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

path_pattern = Path(sys.argv[1]) / r"*Expanded*/*"
for sample in glob.glob(str(path_pattern)):
    sample_folder = Path(sample)
    if not sample.endswith("_0") and not sample.endswith("_1"):
        logging.info(
            f"Skipping {sample_folder.name}, because presumably it is not a t={{0,1}} timepoint."
        )
        continue
    logging.info(f"Computing fragment counts for {sample_folder.name}")
    compute_variant_fragment_size_counts(
        sample_folder, output_folder, path_spreadsheet_runs=run_metadata
    )
logging.info("Done!")
