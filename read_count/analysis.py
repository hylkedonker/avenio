from collections import defaultdict
from typing import Dict, List, Tuple

import pysam


def collect_fragment_sizes(
    bam_file, chromosome: str, start_position: int
) -> Dict[str, List[float]]:
    """
    Pool fragment sizes per base at given position.
    """
    base_counts: Dict[str, List[float]] = defaultdict(list)

    for pile in bam_file.pileup(contig=chromosome, start=start_position):
        # We are only interested in the bases on `start_pos`.
        if pile.pos != start_position:
            continue

        for read in pile.pileups:
            if read.is_del or read.is_refskip:
                continue

            # Skip mates.
            if read.alignment.is_paired and read.alignment.is_read2:
                continue

            base = read.alignment.query_sequence[read.query_position]
            fragment_length = abs(read.alignment.template_length)
            base_counts[base].append(fragment_length)

    return base_counts


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
