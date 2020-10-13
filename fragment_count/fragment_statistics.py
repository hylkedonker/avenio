from collections import defaultdict
from dataclasses import dataclass
import logging
from typing import DefaultDict, List, Tuple

import pysam

from utils import complement, count_fragments, dict_sum


@dataclass
class NucleotideCounts:
    """
    Fragment counts per nucleotide.
    """

    fragment_length: DefaultDict[str, list]
    watson_fourmer: DefaultDict[str, list]
    crick_fourmer: DefaultDict[str, list]
    chromosome: str
    position: int


@dataclass
class NucleotideStatistics(NucleotideCounts):
    """
    Fragment statistics per nucleotide.
    """

    fourmer: DefaultDict[str, list]
    wild_type_base: str
    variant_bases: List[str]


class FragmentStatistics:
    """
    Compute fragment statistics for given BAM file.
    """

    def __init__(self, bam_file: str):
        """
        Load alignment file and build index.
        """
        logging.debug(f"Loading {bam_file}.")
        self.alignments = pysam.AlignmentFile(bam_file)
        logging.debug(f"Building index for {bam_file}.")
        self.read_index = pysam.IndexedReads(self.alignments, multiple_iterators=True)
        self.read_index.build()

    def _determine_wild_type_variant_bases(
        self, fragment_items
    ) -> Tuple[str, List[str]]:
        """ Wild type is most abundant, other bases are variants. """
        base_counts = {len(v): base for base, v in fragment_items.items()}
        normal = base_counts[max(base_counts.keys())]
        variants = [
            base
            for occurs, base in base_counts.items()
            if occurs != 0 and base != normal
        ]
        return normal, variants

    def collect_counts(self, chromosome: str, start_position: int) -> NucleotideCounts:
        """
        Count the fragment occurences per nucleotide.
        """
        counts = NucleotideCounts(
            fragment_length=defaultdict(list),
            watson_fourmer=defaultdict(list),
            crick_fourmer=defaultdict(list),
            chromosome=chromosome,
            position=start_position,
        )

        # Cache all the read names before querying all read pairs, to prevent
        # corrupting the pileup iterator.
        read_name_cache = {}

        for pile in self.alignments.pileup(
            contig=chromosome,
            start=start_position,
            truncate=False,
            ignore_overlaps=True,
        ):
            # We are only interested in the bases on `start_pos`.
            if pile.pos != start_position - 1:
                continue

            for read in pile.pileups:
                if read.is_del or read.is_refskip:
                    continue

                base = read.alignment.query_sequence[read.query_position]
                fragment_length = abs(read.alignment.template_length)
                counts.fragment_length[base].append(fragment_length)
                read_name_cache[read.alignment.query_name] = base

        # Collect both ends of the fragment.
        for query_name, base in read_name_cache.items():
            fragment_pair = list(self.read_index.find(query_name))
            fragment_pair.sort(key=lambda x: x.reference_start)

            if not all(x.reference_name == chromosome for x in fragment_pair):
                logging.warning(
                    f"One or more reads ({query_name}) on wrong chromosome (not on f{chromosome}). Skipping"
                )
                continue
            if len(fragment_pair) > 2:
                logging.warning(f"More than 2 reads for {query_name}. Skipping")
                continue
            if len(fragment_pair) == 1:
                left_read = right_read = None
                singleton_read = fragment_pair[0]
                if singleton_read.is_read1:
                    left_read = singleton_read
                else:
                    right_read = singleton_read
            else:
                left_read, right_read = fragment_pair[:2]

            if left_read:
                watson_fourmer = left_read.query_sequence[:4]
                counts.watson_fourmer[base].append(watson_fourmer)
            if right_read:
                crick_fourmer = complement(right_read.query_sequence[-4:])
                counts.crick_fourmer[base].append(crick_fourmer)

        return counts

    def compute_statistics(self, chromosome: str, start_position: int):
        """
        Calculate statistics for fragments overlapping at given position.
        """
        counts = self.collect_counts(chromosome, start_position)

        all_fourmers = dict_sum(
            counts.watson_fourmer, counts.crick_fourmer, inplace=False
        )

        wild_type, variants = self._determine_wild_type_variant_bases(
            counts.fragment_length
        )

        NucleotideStatistics(
            fragment_length=count_fragments(counts.fragment_length),
            watson_fourmer=count_fragments(counts.watson_fourmer),
            crick_fourmer=count_fragments(counts.crick_fourmer),
            fourmer=count_fragments(all_fourmers),
            chromosome=chromosome,
            position=start_position,
            wild_type_base=wild_type,
            variant_bases=variants,
        )
