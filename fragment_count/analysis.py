import glob
import json
import logging
from pathlib import Path

import pandas as pd

from fragment_statistics import FragmentStatistics


def find_avenio_bam(run_folder: Path) -> Path:
    """
    Find BAM file of AVENIO run.
    """
    folder_name = run_folder.name
    prefix, suffix = folder_name.split("_")

    # Find the bam file.
    bam_pattern = run_folder / f"Deduped-{prefix}*.bam"
    bams = glob.glob(str(bam_pattern))
    assert len(bams) == 1
    bam_file = Path(bams[0])
    return bam_file


def analyse_run_statistics(
    fragment_analyser: FragmentStatistics,
    run_folder: Path,
    output_folder: Path,
    variant_metadata: pd.DataFrame,
):
    """
    Analyse fragment statistics of an Avenio run.
    """
    folder_name = run_folder.name
    prefix, suffix = folder_name.split("_")

    variants = fragment_analyser.compile_variant_statistics(variant_metadata)
    output_json = {
        "patient": prefix,
        "time_point": int(suffix) if suffix.isdigit() else suffix,
        "variants": variants,
    }

    output_file = output_folder / f"{folder_name}.json"
    # To disk.
    with open(output_file, "w") as file_object:
        file_object.write(json.dumps(output_json, indent=4, sort_keys=True))
        logging.debug(
            f"Wrote fragment size counts for {folder_name} to disk:\n {output_file}"
        )


# def pool_variant_fragment_sizes(
#     variant_file, bam_file
# ) -> Tuple[List[float], List[float]]:
#     """
#     Pool fragment sizes for normals vs variants, over all variant calls.
#     """
#     normal_sizes: List[float] = []
#     variant_sizes: List[float] = []

#     for variant in variant_file.fetch():
#         print("Collect fragment sizes for", variant.contig, variant.start, variant.stop)
#         if variant.stop - variant.start != 1:
#             raise NotImplementedError

#         base_counts = collect_fragment_sizes(
#             bam_file, chromosome=variant.contig, start_position=variant.start
#         )

#         # Pool counts for the normal.
#         normals = base_counts[variant.ref.upper()]
#         normal_sizes.extend(normals)
#         # Pool fragment sizes for all variants.
#         for alternat_base in variant.alts:
#             alternats = base_counts[alternat_base.upper()]
#             vaf = len(alternats) / len(normals)
#             print("VAF({})".format(alternat_base), vaf)
#             variant_sizes.extend(alternats)

#     return normal_sizes, variant_sizes
