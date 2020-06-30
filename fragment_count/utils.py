from collections import defaultdict

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
    s = pd.Series(index=range(1, fill_upto_incl + 1), dtype=int)
    keys, values = zip(*distribution.items())
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


def to_cumulative(counts):
    """ Turn count array into cumulative distribution. """
    distribution = counts / sum(counts)
    return distribution.cumsum()
