from collections import defaultdict
from functools import wraps
import glob
import json
import logging
from pathlib import Path
from typing import Any, Callable, List, Tuple

import pandas as pd
import pysam

from fragment_statistics import FragmentStatistics


def compute_variant_fragment_statistics(
    run_folder: Path, output_folder: Path, variant_metadata: pd.DataFrame
):
    """
    Compute the fragment size counts for each variant, and store on disk.
    """
    folder_name = run_folder.name
    prefix, suffix = folder_name.split("_")

    # Find the bam file.
    bam_pattern = run_folder / f"Deduped-{prefix}*.bam"
    bams = glob.glob(str(bam_pattern))
    assert len(bams) == 1
    bam_file = bams[0]

    output_file = output_folder / f"{folder_name}.json"

    analyser = FragmentStatistics(bam_file)

    logging.debug(f"Loading {bam_file}.")
    alignments = pysam.AlignmentFile(bam_file)
    logging.debug(f"Building index for {bam_file}.")
    read_index = pysam.IndexedReads(alignments, multiple_iterators=True)
    read_index.build()

    output_json = {
        "time_point": int(suffix) if suffix.isdigit() else suffix,
        "unparsable_variants": [],
        "variants": [],
    }

    # Go through each variant called by the Avenio platform.
    index_columns = ["Gene", "Genomic Position", "Mutation Class"]
    for idx, row in variant_metadata.iterrows():
        pos = row[index_columns]
        chromosome, position = pos["Genomic Position"].split(":")
        logging.debug(f"Analysing variant {chromosome} at {position}.")

        # Skip CNV's and indels.
        if pos["Mutation Class"] in ("Indel", "CNV"):
            output_json["unparsable_variants"].append(tuple(pos))
            continue
        position = int(position)

        stats = analyser.compute_statistics(chromosome, position)
        variant_item = {
            "chromosome": chromosome,
            "position": position,
            "gene": pos["Gene"],
            "fragment_size_counts": stats["fragment_length"],
            "fourmer_counts": stats["fourmer"],
            "watson_fourmer_counts": stats["watson_fourmer"],
            "crick_fourmer_counts": stats["crick_fourmer"],
        }

        variant_item["fourmer_counts"] = count_fragments(fourmers)

        output_json["variants"].append(variant_item)

    # To disk.
    with open(output_file, "w") as file_object:
        file_object.write(json.dumps(output_json, indent=4, sort_keys=True))
        logging.debug(
            f"Wrote fragment size counts for {folder_name} to disk:\n {output_file}"
        )


def pool_variant_fragment_sizes(
    variant_file, bam_file
) -> Tuple[List[float], List[float]]:
    """
    Pool fragment sizes for normals vs variants, over all variant calls.
    """
    normal_sizes: List[float] = []
    variant_sizes: List[float] = []

    for variant in variant_file.fetch():
        print("Collect fragment sizes for", variant.contig, variant.start, variant.stop)
        if variant.stop - variant.start != 1:
            raise NotImplementedError

        base_counts = collect_fragment_sizes(
            bam_file, chromosome=variant.contig, start_position=variant.start
        )

        # Pool counts for the normal.
        normals = base_counts[variant.ref.upper()]
        normal_sizes.extend(normals)
        # Pool fragment sizes for all variants.
        for alternat_base in variant.alts:
            alternats = base_counts[alternat_base.upper()]
            vaf = len(alternats) / len(normals)
            print("VAF({})".format(alternat_base), vaf)
            variant_sizes.extend(alternats)

    return normal_sizes, variant_sizes
