import glob
import json
import logging
from pathlib import Path

import pandas as pd

from fragment_statistics import FragmentStatistics


def analyse_run_statistics(
    run_folder: Path, output_folder: Path, variant_metadata: pd.DataFrame
):
    """
    Analyse fragment statistics of an Avenio run.
    """
    folder_name = run_folder.name
    prefix, suffix = folder_name.split("_")

    # Find the bam file.
    bam_pattern = run_folder / f"Deduped-{prefix}*.bam"
    bams = glob.glob(str(bam_pattern))
    assert len(bams) == 1
    bam_file = Path(bams[0])

    variants = compute_variant_fragment_statistics(bam_file, variant_metadata)
    output_json = {
        "patient": prefix,
        "time_point": int(suffix) if suffix.isdigit() else suffix,
        # "unparsable_variants": [],
        "variants": variants,
    }

    output_file = output_folder / f"{folder_name}.json"
    # To disk.
    with open(output_file, "w") as file_object:
        file_object.write(json.dumps(output_json, indent=4, sort_keys=True))
        logging.debug(
            f"Wrote fragment size counts for {folder_name} to disk:\n {output_file}"
        )


def compute_variant_fragment_statistics(
    bam_file: Path, variant_metadata: pd.DataFrame
) -> list:
    """
    Compute the fragment size counts for each variant, and store on disk.
    """
    analyser = FragmentStatistics(bam_file)

    required_columns = ["Gene", "Genomic Position", "Mutation Class"]
    assert set(required_columns).issubset(set(variant_metadata.columns))

    variant_statistics = []
    # Go through each variant called by the Avenio platform.
    for idx, row in variant_metadata.iterrows():
        chromosome, position = row["Genomic Position"].split(":")
        logging.debug(f"Analysing variant {chromosome} at {position}.")

        # Skip CNV's and indels.
        if row["Mutation Class"] in ("Indel", "CNV"):
            logging.debug("Skipping non-SNV variant")
            logging.debug(row)
            continue
        position = int(position)

        stats = analyser.compute_statistics(chromosome, position)
        stats["gene"] = row["Gene"]
        variant_statistics.append(stats)
    return variant_statistics


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
