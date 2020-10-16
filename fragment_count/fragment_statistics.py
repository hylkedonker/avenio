from collections import defaultdict
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import DefaultDict, List, Tuple, Union

import pandas as pd
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


class FragmentStatistics:
    """
    Compute fragment statistics for given BAM file.
    """

    def __init__(self, bam_file: Union[str, Path]):
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

            # Determine ordering of read pair wrt reference genome.
            if len(fragment_pair) > 2:
                logging.warning(f"More than 2 reads for {query_name}. Skipping")
                continue
            if len(fragment_pair) == 1:
                left_read = right_read = None
                singleton_read = fragment_pair[0]
                if singleton_read.pos < singleton_read.next_reference_start:
                    left_read = singleton_read
                else:
                    right_read = singleton_read
            else:
                left_read, right_read = fragment_pair[:2]

            # Extract fourmer.
            if left_read:
                watson_fourmer = left_read.query_sequence[:4]
                counts.watson_fourmer[base].append(watson_fourmer)
            # Take complement, see Fig. 1 Jiang et al., Cancer Discovery 10, 665 ('20).
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

        return {
            "fragment_length": count_fragments(counts.fragment_length),
            "watson_fourmer": count_fragments(counts.watson_fourmer),
            "crick_fourmer": count_fragments(counts.crick_fourmer),
            "fourmer": count_fragments(all_fourmers),
            "chromosome": chromosome,
            "position": start_position,
            "wild_type_base": wild_type,
            "variant_bases": variants,
        }

    def compile_variant_statistics(self, variant_metadata: pd.DataFrame) -> list:
        """
        Compute the fragment size statistics for each variant in `variant_metadata`.
        """

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

            stats = self.compute_statistics(chromosome, position)
            stats["gene"] = row["Gene"]
            variant_statistics.append(stats)
        return variant_statistics
