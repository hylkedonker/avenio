from collections import Counter, defaultdict
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pysam


def collect_fragment_sizes(
    bam_file, chromosome: str, start_position: int
) -> Dict[str, List[float]]:
    """
    Pool fragment sizes per base at given position.
    """
    base_counts: Dict[str, List[float]] = defaultdict(list)

    for pile in bam_file.pileup(
        contig=chromosome, start=start_position, truncate=False
    ):
        # We are only interested in the bases on `start_pos`.
        if pile.pos != start_position - 1:
            # print("no pos match")
            continue

        for read in pile.pileups:
            if read.is_del or read.is_refskip:
                continue

            # # Skip mates, because they refer to the same fragment.
            # if read.alignment.is_paired and read.alignment.is_read2:
            #     continue

            base = read.alignment.query_sequence[read.query_position]
            fragment_length = abs(read.alignment.template_length)
            base_counts[base].append(fragment_length)

    return base_counts


def compute_variant_fragment_size_counts(
    run_folder: Path, output_folder: Path, path_spreadsheet_runs: Path
):
    """
    Compute the fragment size counts for each variant, and store on disk.
    """
    folder_name = run_folder.name
    bam_file = run_folder / f"Deduped-{folder_name}.bam"
    output_file = output_folder / f"{folder_name}.json"

    alignments = pysam.AlignmentFile(bam_file)
    logging.debug(f"Loading {bam_file}.")

    # Find variants for this run from the spreadsheet file.
    run_sheet = pd.read_excel(path_spreadsheet_runs, sheet_name=1)
    columns_to_keep = ["Gene", "Coding Change", "Genomic Position"]
    index_columns = ["Gene", "Genomic Position"]
    run_variants = run_sheet[run_sheet["Sample ID"] == folder_name]
    run_variants = run_variants[columns_to_keep]

    output_json = {
        "time_point": int(folder_name.split("_")[1]),
        "unparsable_variants": [],
        "variants": [],
    }

    # Go through each variant.
    for idx, row in run_variants.iterrows():
        pos = row[index_columns]
        chromosome, position = pos["Genomic Position"].split(":")
        logging.debug(f"Analysing variant {chromosome} at {position}.")

        # Skip CNV's.
        if "-" in position:
            output_json["unparsable_variants"].append(tuple(pos))
            continue

        variant_item = {
            "chromosome": chromosome,
            "position": position,
            "gene": pos["Gene"],
            "fragment_size_counts": {"A": {}, "C": {}, "T": {}, "G": {}},
        }

        fragment_sizes = collect_fragment_sizes(alignments, chromosome, int(position))
        fragment_counts = {len(v): k for k, v in fragment_sizes.items()}
        variant_item["nucleotide_normal"] = fragment_counts[max(fragment_counts.keys())]
        variant_item["nucleotide_variants"] = [
            v
            for k, v in fragment_counts.items()
            if k != 0 and v != variant_item["nucleotide_normal"]
        ]
        for base in fragment_sizes.keys():
            # Compute the number of occurences of each fragment size.
            counts = Counter(fragment_sizes[base])
            variant_item["fragment_size_counts"][base] = counts

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
