from collections import defaultdict
from pprint import pprint
import re
from typing import Optional

import numpy as np
import pandas as pd
import requests

from const import target_genes


def filter_pathways(gene_pathways):
    """
    Filter out non-interesting pathways.
    """
    # Select only cancer related pathways, if possible.
    cancer_related = gene_pathways["Entry"].str.contains("hsa052")
    cancer_pathways = gene_pathways[cancer_related]
    if len(cancer_pathways) > 0:
        return cancer_pathways

    # Otherwise try to find immune related pathways.
    immune_related = gene_pathways["Entry"].str.contains("hsa053")
    immune_pathways = gene_pathways[immune_related]
    if len(immune_pathways) > 0:
        return immune_pathways

    # Otherwise don't filter.
    return gene_pathways


def find_gene(gene_name: str) -> Optional[str]:
    """
    Find KEGG entry name.
    """
    url = f"http://rest.kegg.jp/find/hsa/{gene_name}"
    webpage = requests.get(url).text
    for line in webpage.splitlines():
        columns = line.split("\t")
        kegg_id = columns[0]

        # No associated gene to parse, skip.
        if ";" not in columns[1]:
            continue
        synonyms, description = columns[1].split(";")

        if re.search(
            pattern=f"(\A|\b){gene_name}(,|\b| )+", string=synonyms, flags=re.IGNORECASE
        ):
            if "hsa" not in kegg_id:
                raise ValueError(f"Problem finding gene {gene_name} @ {url}.")
            return kegg_id


def find_networks(kegg_id: str, pattern="nt062") -> dict:
    """
    Find cancer related networks for a gene using KEGG id.
    """
    networks = {}
    url = f"http://rest.kegg.jp/get/{kegg_id}"
    webpage = requests.get(url).text
    for line in webpage.splitlines():
        columns = line.strip().split("  ")
        # Look for networks matching the pattern (nt062 refers to cancer related
        # networks).
        if any(c.strip().startswith(pattern) for c in columns):
            networks[columns[-2].strip()] = columns[-1].strip()
    return networks


def fetch_networks(output_filename) -> pd.DataFrame:
    """
    Download network annotations from KEGG.
    """
    network_map = pd.DataFrame(
        "not_annotated", index=target_genes, columns=["entries", "names"]
    )
    for gene in target_genes:
        if (kegg_id := find_gene(gene)) :
            if (gene_networks := find_networks(kegg_id)) :
                network_map.loc[gene, ["entries", "names"]] = [
                    ",".join(gene_networks.keys()),
                    "|".join(gene_networks.values()),
                ]
            print(network_map.loc[gene])
            print("--")
        else:
            print(f"No networks found for {gene}.")

    network_map.to_excel(output_filename, sheet_name="AVENIO Gene <==> Network")

    return network_map


def fetch_pathways(output_filename):
    """
    Download pathway annotations from KEGG.
    """
    pathway_map = pd.DataFrame(index=target_genes, columns=["entries", "names"])
    for gene in target_genes:
        url = f"https://www.kegg.jp/kegg-bin/search_pathway_text?map=hsa&keyword={gene}&mode=1"
        df = pd.read_html(url)[1]
        if len(df) == 0 or df.shape == (1, 1):
            print(f"No hits for {gene}!")
            pathway_map.loc[gene] = ["not_annotated"]
            continue

        df = filter_pathways(df)

        pathway_map.loc[gene, ["entries", "names"]] = [
            ",".join(df["Entry"]),
            "|".join(df["Name"]),
        ]
        print(gene, ":\n", pathway_map.loc[gene])
    pathway_map.to_excel(output_filename, sheet_name="AVENIO Gene <==> Pathway")

    return pathway_map


def assign_most_common_pathway(input_filename, output_filename):
    """
    From all pathways in which a gene is involved, select the most frequently occuring.
    """
    df = pd.read_excel(input_filename)
    df.set_index(df.columns[0], inplace=True)
    top_pathway = pd.DataFrame(index=df.index, columns=["network", "name"])

    # Count occurences of each pathway.
    pathway_counts = defaultdict(int)
    pathway_names = {}
    df["entries"] = df["entries"].apply(
        lambda x: x.split(",") if isinstance(x, str) else []
    )
    df["names"] = df["names"].apply(
        lambda x: x.split("|") if isinstance(x, str) else []
    )
    for entries, names in zip(df["entries"], df["names"]):
        for pathway, name in zip(entries, names):
            pathway_names[pathway] = name
            pathway_counts[pathway] += 1

    def select_most_common_pathway(entries_column):
        entry_occurences = tuple(pathway_counts[e] for e in entries_column)
        i = np.argmax(entry_occurences)
        return entries_column[i]

    # For each gene, assign pathway with most genes.
    top_pathway["network"] = df["entries"].apply(select_most_common_pathway)
    top_pathway["name"] = top_pathway["network"].map(pathway_names)
    top_pathway.to_excel(output_filename)


# fetch_pathways(output_filename="/tmp/gene_pathway_all.xlsx")
network_map = fetch_networks(output_filename="/tmp/gene_networks_all.xlsx")
# pprint(network_map)

assign_most_common_pathway(
    input_filename="/tmp/gene_networks_all.xlsx",
    output_filename="gene_annotation.xlsx",
)
