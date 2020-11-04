from collections import defaultdict, Counter
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

from numpy import log
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

    for base in b.keys():
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
    index_cache = set()
    gene_series = {}
    genes = sorted(gene_counts.keys())
    for gene in genes:
        s = dict_as_series(gene_counts[gene], index)
        gene_series[gene] = s
        index_cache = index_cache.union(set(s.index))

    reindex = list(index_cache)
    reindex.sort()

    if index is not None and len(index) > len(reindex):
        reindex = index

    df = pd.DataFrame(gene_series, index=reindex, columns=genes).fillna(0).astype(int)

    is_digit = map(
        lambda x: True if isinstance(x, str) and x.isdigit() else False, df.index
    )

    if all(is_digit):
        reindex = list(map(int, df.index))
        df.index = reindex
        return df.sort_index()

    return df


def complement(strand):
    translation = {"A": "T", "C": "G", "T": "A", "G": "C"}
    cstrand = map(lambda x: translation[x], strand)
    return "".join(cstrand)


def json_to_frame(filename: str, field_name: str):
    """
    Load single JSON file into pandas dataframe.
    """
    normal_counts = defaultdict(lambda: defaultdict(int))
    variant_counts = defaultdict(lambda: defaultdict(int))

    with open(filename) as file_object:
        fragments = json.load(file_object)
        for var in fragments["variants"]:
            normal_base = var["wild_type_base"]
            gene = var["gene"]

            # Combine Watson and Crick fourmers into single field.
            if field_name == "fourmer":
                counts = defaultdict(lambda: defaultdict(int))
                counts = dict_sum_sum(counts, var["watson_fourmer"])
                counts = dict_sum_sum(counts, var["crick_fourmer"])
            else:
                counts = var[field_name]
            normal_counts[gene] = dict_sum(normal_counts[gene], counts[normal_base])
            for base in var["variant_bases"]:
                variant_counts[gene] = dict_sum(variant_counts[gene], counts[base])

    index = None
    if field_name in ("fourmer", "watson_fourmer", "crick_fourmer"):
        index = pd.Series(range(4 ** 4)).apply(int_to_fourmer)
    normals = dict_to_frame(normal_counts, index)
    variants = dict_to_frame(variant_counts, index)
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
    if field_name in ("watson_fourmer", "crick_fourmer", "fourmer"):
        for df in data_frames:
            genes = genes.union(set(df.columns))
        # Index are all possible fourmers.
        index = pd.Series(range(4 ** 4)).apply(int_to_fourmer)
        return index, genes

    elif field_name == "fragment_length":
        max_fragment_size = 1
        for df in data_frames:
            if not df.empty:
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
        if field_name == "fourmer":
            item_name = "4mer"
        elif field_name == "watson_fourmer":
            item_name = "watson 4mer"
        elif field_name == "crick_fourmer":
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


def _get_principle_axis(frame):
    """ Determine which is the primary column of interest. """
    principle_axis = "length (bp)"
    potential_columns = [
        c for c in frame.reset_index().columns if isinstance(c, str) and "4mer" in c
    ]
    if any(potential_columns):
        principle_axis = potential_columns[0]
    return principle_axis


def pool(df, normalise=False):
    """ Pool over genes, patients, and samples. """
    principle_axis = _get_principle_axis(df)
    pooled_series = df.sum(axis=1).groupby(principle_axis).sum().astype(int)
    if normalise:
        return safe_normalise(pooled_series)
    return pooled_series


def pool_timepoints(data_frame):
    """ Combine baseline and follow up samples. """
    principle_axis = _get_principle_axis(data_frame)
    return data_frame.groupby(["Patient ID", principle_axis]).sum()


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


def entropy(p):
    S = -p * log(p, where=p != 0.0) / log(256)
    return S.sum()


def select_sample_variants(
    sample_folder_name: str,
    metadata_location=Path("/metadata/variant_list_20200730.xlsx"),
) -> pd.DataFrame:
    """ Find variants for this run from the spreadsheet run file. """
    patient_id, sample_type = sample_folder_name.split("_")
    patient_id = int(patient_id)

    sheets = pd.read_excel(metadata_location, sheet_name=[1, 2, 3])
    tumor_sheet = sheets[1]
    pbmc_plasma_sheet = sheets[2]
    chip_sheet = sheets[3]

    columns_to_keep = ["Gene", "Coding Change", "Genomic Position", "Mutation Class"]
    if "PBMC" in sample_folder_name:
        # Keep PBMC variants from both timepoints. We will use `drop_duplicates` below
        # for variants that are both present.
        pbmc_constraint = pbmc_plasma_sheet["Patient ID"] == patient_id
        return pbmc_plasma_sheet[pbmc_constraint].drop_duplicates().copy()

    tumor_constraint = tumor_sheet["Sample ID"] == sample_folder_name
    chip_constraint = chip_sheet["Sample ID"] == sample_folder_name
    # Select the genomic position from the pbmc-plasma sheet, because it is missing in
    # the chip sheet.
    tumors = tumor_sheet[tumor_constraint]
    chips = chip_sheet[chip_constraint].merge(
        pbmc_plasma_sheet[pbmc_plasma_sheet["Sample ID"] == sample_folder_name],
        on=["Gene", "Coding Change", "Mutation Class"],
        how="inner",
        suffixes=("", "_y"),
    )

    return (
        tumors.filter(items=columns_to_keep).drop_duplicates().copy(),
        chips.filter(items=columns_to_keep).drop_duplicates().copy(),
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
