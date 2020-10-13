from collections import Counter, defaultdict
from functools import wraps
import glob
import json
import logging
from pathlib import Path
from typing import Any, Callable, List, Tuple

import pandas as pd
import pysam

from utils import complement


def from_pileup(function_object: Callable[..., Tuple[str, Any]]):
    """
    Wrapper that iterates through the reads piling up at a specific genomic position.
    """

    @wraps(function_object)
    def wrapped_read_collection(bam_file, chromosome: str, start_position: int):
        """
        Iterate through reads at genomic position, and collect key-value pairs.
        """
        # The items (returned as Key-value pairs) are collected in this container.
        item_container = defaultdict(list)
        for pile in bam_file.pileup(
            contig=chromosome,
            start=start_position,
            truncate=False,
            ignore_overlaps=True,
        ):
            # We are only interested in the bases on `start_pos`.
            if pile.pos != start_position - 1:
                # print("no pos match")
                continue

            for read in pile.pileups:
                if read.is_del or read.is_refskip:
                    continue
                item_name, item_value = function_object(read)
                item_container[item_name].append(item_value)

        return item_container

    return wrapped_read_collection


@from_pileup
def collect_fragment_sizes(read) -> Tuple[str, int]:
    """
    Extract fragment size from read.
    """
    base = read.alignment.query_sequence[read.query_position]
    fragment_length = abs(read.alignment.template_length)
    return base, fragment_length


def collect_fragment_fourmer(bam_file, chromosome: str, start_position: int):
    # The items (returned as Key-value pairs) are collected in this container.
    item_container = defaultdict(list)
    for pile in bam_file.pileup(
        contig=chromosome, start=start_position, truncate=False, ignore_overlaps=True
    ):
        # We are only interested in the bases on `start_pos`.
        if pile.pos != start_position - 1:
            # print("no pos match")
            continue

        for read in pile.pileups:
            if read.is_del or read.is_refskip:
                continue

            base = read.alignment.query_sequence[read.query_position]

            # Save location for iterator, because jumping to mate may change location of
            # iterator (see PySam documentation).
            pointer = bam_file.tell()

            try:
                read_mate = bam_file.mate(read.alignment)
            except ValueError:
                print("No mate for. ", read.alignment.query_name)
                continue

            if read.alignment.reference_start < read_mate.reference_start:
                left_read, right_read = read.alignment, read_mate
            else:
                left_read, right_read = read_mate, read.alignment

            watson_fourmer = left_read.query_sequence[:4]
            crick_fourmer = complement(right_read.query_sequence[-4:])
            item_container[base].extend([watson_fourmer, crick_fourmer])

            bam_file.seek(pointer)

    return item_container


@from_pileup
def collect_fragment_watson_fourmer(read) -> Tuple[str, str]:
    """
    Extract fragment fourmer from 5' end motif Watson strand.
    """
    base = read.alignment.query_sequence[read.query_position]
    # Leading fourmer.
    four_mer = read.alignment.query_sequence[:4]
    assert four_mer.isupper()
    return base, four_mer


@from_pileup
def collect_fragment_crick_fourmer(read) -> Tuple[str, str]:
    """
    Extract fragment fourmer from 5' end motif Crick strand.
    """
    base = read.alignment.query_sequence[read.query_position]
    # Trailing fourmer, mirrored to obtain Crick strand.
    four_mer = read.alignment.query_sequence[-4:]
    assert four_mer.isupper()
    return base, complement(four_mer)


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

    alignments = pysam.AlignmentFile(bam_file)
    logging.debug(f"Loading {bam_file}.")

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

        variant_item = {
            "chromosome": chromosome,
            "position": position,
            "gene": pos["Gene"],
            "fragment_size_counts": {},
        }
        fragment_sizes = collect_fragment_sizes(alignments, chromosome, position)
        fourmers = collect_fragment_fourmer(alignments, chromosome, position)
        watson_fourmers = collect_fragment_watson_fourmer(
            alignments, chromosome, position
        )
        crick_fourmers = collect_fragment_crick_fourmer(
            alignments, chromosome, position
        )

        wild_type, variants = _get_wild_type_and_variant_nucleotides(fragment_sizes)
        variant_item["nucleotide_normal"] = wild_type
        variant_item["nucleotide_variants"] = variants
        variant_item["fragment_size_counts"] = count_fragments(fragment_sizes)
        variant_item["watson_fourmer_counts"] = count_fragments(watson_fourmers)
        variant_item["crick_fourmer_counts"] = count_fragments(crick_fourmers)
        variant_item["fourmer_counts"] = count_fragments(fourmers)

        output_json["variants"].append(variant_item)

    # To disk.
    with open(output_file, "w") as file_object:
        file_object.write(json.dumps(output_json, indent=4, sort_keys=True))
        logging.debug(
            f"Wrote fragment size counts for {folder_name} to disk:\n {output_file}"
        )


def count_fragments(fragment_items) -> dict:
    """
    Compute the number of occurences of each fragment item (e.g., size, or motif).
    """
    counts_per_base = {}
    for base in fragment_items.keys():
        counts = Counter(fragment_items[base])
        counts_per_base[base] = counts

    return counts_per_base


def _get_wild_type_and_variant_nucleotides(fragment_items) -> Tuple[str, List[str]]:
    """ Wild type is most abundant, other bases are variants. """
    base_counts = {len(v): base for base, v in fragment_items.items()}
    normal = base_counts[max(base_counts.keys())]
    variants = [
        base for occurs, base in base_counts.items() if occurs != 0 and base != normal
    ]
    return normal, variants


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
