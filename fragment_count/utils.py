from collections import defaultdict
import json
from typing import Dict, Iterable, Tuple

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
        result[key] += value
    return result


def dict_sum_sum(
    a: Dict[str, dict], b: Dict[str, dict], inplace: bool = True
) -> defaultdict:
    """
    Perform dict_sum, key-by-key.
    """
    if not inplace:
        result = a.copy()
    else:
        result = a

    for base in set(a.keys()).union(b.keys()):
        result[base] = dict_sum(defaultdict(int, a[base]), b[base], inplace)
    return result


def dict_as_series(distribution: dict, index=None) -> pd.Series:
    """
    Turn count dictionary into series.
    """
    if index is None:
        s = pd.Series(distribution, dtype=int)
        return s.sort_index()
    keys, values = zip(*distribution.items())
    s = pd.Series(index=index, dtype=int)
    if len(keys) > 1:
        s[keys] = values
    else:
        # Hack for pandas bug.
        s[keys[0]] = values[0]
    return s


def dict_to_frame(gene_counts: defaultdict, index=None) -> pd.DataFrame:
    """
    Combine all distributions in `gene_counts` into a data frame.
    """
    genes = sorted(gene_counts.keys())
    df = pd.DataFrame(columns=genes)
    if index is not None:
        df.set_index(index)

    for gene in genes:
        df[gene] = dict_as_series(gene_counts[gene], index)
    return df.fillna(0).astype(int)


def json_to_frame(filename: str, field_name: str):
    """
    Load single JSON file into pandas dataframe.
    """
    normal_counts = defaultdict(lambda: defaultdict(int))
    variant_counts = defaultdict(lambda: defaultdict(int))

    with open(filename) as file_object:
        fragments = json.load(file_object)
        for var in fragments["variants"]:
            # Combine Watson and Crick fourmers into single field.
            if field_name == "fourmer_counts":
                counts = defaultdict(lambda: defaultdict(int))
                counts = dict_sum_sum(counts, var["watson_fourmer_counts"])
                counts = dict_sum_sum(counts, var["crick_fourmer_counts"])
                raise
            else:
                counts = var[field_name]
            normal_base = var["nucleotide_normal"]
            gene = var["gene"]
            normal_counts[gene] = dict_sum(normal_counts[gene], counts[normal_base])
            for base in var["nucleotide_variants"]:
                variant_counts[gene] = dict_sum(variant_counts[gene], counts[base])

    import ipdb

    ipdb.set_trace()
    normals = dict_to_frame(normal_counts)
    variants = dict_to_frame(variant_counts)
    return normals, variants


def int_to_fourmer(number: int) -> str:
    """ Map binary coding of integer to bases. """
    base_map = {0: "A", 1: "C", 2: "T", 3: "G"}
    fourmer = ""
    for i in range(4):
        ith_int = (number >> (2 * i)) & 3
        base = base_map[ith_int]
        fourmer += base
    return fourmer[::-1]


def compute_domain(data_frames, field_name: str):
    """
    Determine the largest indices (fragment size) and columns (gene names).
    """
    genes = set()
    if field_name in (
        "watson_fourmer_counts",
        "crick_fourmer_counts",
        "fourmer_counts",
    ):
        for df in data_frames:
            genes = genes.union(set(df.columns))
        # Index are all possible fourmers.
        index = pd.Series(range(4 ** 4)).apply(int_to_fourmer)
        return index, genes

    elif field_name == "fragment_size_counts":
        max_fragment_size = 1
        for df in data_frames:
            max_fragment_size = max(max_fragment_size, max(df.index.astype(int)))
            genes = genes.union(set(df.columns))
        return range(1, max_fragment_size + 1), sorted(genes)
    raise NotImplementedError


def pool_counts_to_data_frame(json_files) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate count statistics from list of json files.
    """
    normals, variants = load_samples_as_frame(json_files)
    return normals.groupby("length (bp)").sum(), variants.groupby("length (bp)").sum()


def load_samples_as_frame(filenames: Iterable[str], field_name: str) -> pd.DataFrame:
    """
    Concatenate sample data frames in one monolithic multi-index data frame.
    """
    # Cache data frames per sample.
    normal_variant_pairs = [json_to_frame(f, field_name) for f in filenames]
    normals, variants = zip(*normal_variant_pairs)
    # Determine domain of the samples.
    indices, columns = compute_domain(tuple(normals) + tuple(variants), field_name)
    kwargs = {"index": indices, "columns": columns, "dtype": int}

    def _concatenate(data_frames):
        items = [pd.DataFrame(0, **kwargs).add(df, fill_value=0) for df in data_frames]
        names = [tuple(f.split("/")[-1].split(".")[0].split("_")) for f in filenames]
        item_name = "length (bp)"
        if field_name == "fourmer_counts":
            item_name = "4mer"
        elif field_name == "watson_fourmer_counts":
            item_name = "watson 4mer"
        elif field_name == "crick_fourmer_counts":
            item_name = "crick 4mer"
        panel = pd.concat(
            items, keys=names, names=["Patient ID", "sample number", item_name], axis=0
        ).astype(int)
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
