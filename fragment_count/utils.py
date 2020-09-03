from collections import defaultdict
import json
from typing import Iterable, Tuple

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


def dict_as_series(distribution: dict, fill_upto_incl=-1) -> pd.Series:
    """
    Turn count dictionary into series.
    """
    if fill_upto_incl == -1:
        s = pd.Series(distribution)
        return s.sort_index()
    keys, values = zip(*distribution.items())
    s = pd.Series(index=range(1, max(max(keys), fill_upto_incl) + 1), dtype=int)
    if len(keys) > 1:
        s[keys] = values
    else:
        # Hack for pandas bug.
        s[keys[0]] = values[0]
    return s


def dict_to_data_frame(
    gene_counts: defaultdict, max_fragment_size: int
) -> pd.DataFrame:
    """
    Combine all distributions in `gene_counts` into a data frame.
    """
    genes = sorted(gene_counts.keys())
    df = pd.DataFrame(index=range(1, max_fragment_size + 1), columns=genes)
    for gene in genes:
        df[gene] = dict_as_series(gene_counts[gene], fill_upto_incl=max_fragment_size)
    return df


def _json_to_data_frame(filename: str):
    """
    Load single JSON file into pandas dataframe.
    """
    normal_counts = defaultdict(lambda: defaultdict(int))
    variant_counts = defaultdict(lambda: defaultdict(int))
    max_fragment_size = 1
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
    normals = dict_to_data_frame(normal_counts, max_fragment_size)
    variants = dict_to_data_frame(variant_counts, max_fragment_size)
    return normals, variants


def compute_domain(data_frames):
    """
    Determine the largest indices (fragment size) and columns (gene names).
    """
    genes = set()
    max_fragment_size = 1
    for df in data_frames:
        max_fragment_size = max(max_fragment_size, max(df.index))
        genes = genes.union(set(df.columns))
    return range(1, max_fragment_size + 1), sorted(genes)


def pool_counts_to_data_frame(json_files) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate count statistics from list of json files.
    """
    normals, variants = load_samples_as_data_frame(json_files)
    return normals.groupby("length (bp)").sum(), variants.groupby("length (bp)").sum()


def load_samples_as_data_frame(filenames: Iterable[str]) -> pd.DataFrame:
    """
    Concatenate sample data frames in one monolithic multi-index data frame.
    """
    # Cache data frames per sample.
    normal_variant_pairs = [_json_to_data_frame(f) for f in filenames]
    normals, variants = zip(*normal_variant_pairs)
    # Determine domain of the samples.
    indices, columns = compute_domain(tuple(normals) + tuple(variants))
    kwargs = {"index": indices, "columns": columns, "dtype": int}

    def _concatenate(data_frames):
        items = [pd.DataFrame(0, **kwargs).add(df, fill_value=0) for df in data_frames]
        names = [tuple(f.split("/")[-1].split(".")[0].split("_")) for f in filenames]
        panel = pd.concat(
            items,
            keys=names,
            names=["Patient ID", "sample number", "length (bp)"],
            axis=0,
        )
        return panel

    return _concatenate(normals), _concatenate(variants)


def to_cumulative(counts):
    """ Turn count array into cumulative distribution. """
    distribution = counts / sum(counts)
    return distribution.cumsum()


def pool(df):
    """ Pool over genes, patients, and samples. """
    return df.sum(axis=1).groupby("length (bp)").sum().astype(int)


def pool_timepoints(data_frame):
    """ Combine baseline and follow up samples. """
    return data_frame.groupby(["Patient ID", "length (bp)"]).sum()


def safe_normalise(x):
    """ Turn counts into normalised distribution. """
    Z = x.sum()
    if Z != 0.0:
        return x / Z
    return x


def filter_no_fragments(data_frame):
    """ Remove patients with zero fragment count, over all genes. """
    clearance = data_frame.groupby("Patient ID").sum() == 0
    return data_frame[
        ~clearance[data_frame.index.get_level_values("Patient ID")].values
    ]


def pool_and_normalise(data_frame):
    """ Pool over time point and genes. """
    return (
        pool_timepoints(data_frame)
        .sum(axis=1)
        .groupby("Patient ID")
        .apply(safe_normalise)
    )
