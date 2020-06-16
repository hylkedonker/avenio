from collections import Counter, defaultdict
import os
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
    print(chromosome, start_position)

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

            # Skip mates, because they refer to the same fragment.
            if read.alignment.is_paired and read.alignment.is_read2:
                continue

            base = read.alignment.query_sequence[read.query_position]
            fragment_length = abs(read.alignment.template_length)
            base_counts[base].append(fragment_length)

    return base_counts


def enlarge_data_frame(data_frame, new_index_size):
    """
    Expand the number of rows up to and including `index_size`.
    """
    empty_df = pd.DataFrame(index=range(data_frame.index[-1], new_index_size + 1))
    return data_frame.append(empty_df).fillna(0)


def compute_variant_fragment_size_counts(
    run_folder: Path, output_folder: Path, run_spreadsheet_path: Path
):
    """
    Compute the fragment size counts for each variant, and store on disk.
    """
    folder_name = run_folder.name
    bam_file = run_folder / f"Deduped-{folder_name}.bam"
    output_csv = output_folder / f"{folder_name}__fragment_size_counts.csv"
    no_output_csv = output_folder / f"{folder_name}__unparsable.csv"

    alignments = pysam.AlignmentFile(bam_file)

    # Find variants for this run from the spreadsheet file.
    run_sheet = pd.read_excel(run_spreadsheet_path, sheet_name=1)
    columns_to_keep = ["Gene", "Coding Change", "Genomic Position"]
    index_columns = ["Gene", "Genomic Position"]
    run_variants = run_sheet[run_sheet["Sample ID"] == folder_name]
    run_variants = run_variants[columns_to_keep]

    variants_unparsable = pd.DataFrame(index=run_variants[index_columns])
    size_counts = pd.DataFrame(index=range(1, 500))

    # Go through each variant.
    for idx, row in run_variants.iterrows():
        pos = row[index_columns]
        chromosome, position = pos["Genomic Position"].split(":")

        # Skip CNV's.
        if "-" in position:
            continue
        # Remove variant from the unparsable list.
        variants_unparsable.drop(index=tuple(pos), inplace=True)

        fragment_sizes = collect_fragment_sizes(alignments, chromosome, int(position))
        for base in fragment_sizes.keys():
            column_name = tuple(pos) + (base,)
            size_counts[column_name] = 0
            # Compute the number of occurences of each fragment size.
            counts = Counter(fragment_sizes[base])

            # Enlarge index, when a fragment size falls outside the index range.
            largest_fragment = max(counts.keys())
            if largest_fragment > size_counts.index[-1]:
                size_counts = enlarge_data_frame(
                    size_counts, new_index_size=largest_fragment
                )

            ordered_keys, ordered_values = zip(*counts.items())
            size_counts.loc[ordered_keys, column_name] = ordered_values

    size_counts.to_csv(output_csv)
    variants_unparsable.to_csv(no_output_csv)


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
