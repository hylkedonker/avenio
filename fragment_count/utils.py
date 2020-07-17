from collections import defaultdict
import json
from typing import Tuple

import pandas as pd


def dict_sum(a: defaultdict, b: dict, inplace: bool = True) -> defaultdict:
    """
    Calculate a + b, key-by-key.
    """
    if not inplace:
        result = a.copy()
    else:
        result = a
    for key, value in b.items():
        result[int(key)] += value
    return result


def as_series(distribution: dict, fill_upto_incl=-1) -> pd.Series:
    """
    Turn count dictionary into series.
    """
    if fill_upto_incl == -1:
        s = pd.Series(distribution)
        return s.sort_index()
    keys, values = zip(*distribution.items())
    s = pd.Series(index=range(1, max(max(keys), fill_upto_incl) + 1), dtype=int)
    s[keys] = values
    return s


def as_dataframe(gene_counts: defaultdict, max_fragment_size: int) -> pd.DataFrame:
    """
    Combine all distributions in `gene_counts` into a data frame.
    """
    genes = sorted(gene_counts.keys())
    df = pd.DataFrame(index=range(1, max_fragment_size + 1), columns=genes)
    for gene in genes:
        df[gene] = as_series(gene_counts[gene], fill_upto_incl=max_fragment_size)
    return df


def pool_counts_to_dataframe(json_files) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate count statistics from list of json files.
    """
    normal_counts = defaultdict(lambda: defaultdict(int))
    variant_counts = defaultdict(lambda: defaultdict(int))
    max_fragment_size = 1
    for filename in json_files:
        with open(filename) as file_object:
            fragments = json.load(file_object)
            for var in fragments["variants"]:
                counts = var["fragment_size_counts"]
                normal_base = var["nucleotide_normal"]
                gene = var["gene"]
                normal_counts[gene] = dict_sum(normal_counts[gene], counts[normal_base])
                for base in var["nucleotide_variants"]:
                    variant_counts[gene] = dict_sum(variant_counts[gene], counts[base])
            max_fragment_size = max(
                max(normal_counts[gene].keys()),
                max(variant_counts[gene].keys()),
                max_fragment_size,
            )

    normals = as_dataframe(normal_counts, max_fragment_size)
    variants = as_dataframe(variant_counts, max_fragment_size)
    return normals, variants


def to_cumulative(counts):
    """ Turn count array into cumulative distribution. """
    distribution = counts / sum(counts)
    return distribution.cumsum()
