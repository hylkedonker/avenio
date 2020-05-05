from collections import defaultdict

import numpy as np
import pandas as pd

from const import target_genes


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
            pathway_map.loc[gene] = ["None"]
            continue

        pathway_map.loc[gene, ["entries", "names"]] = [
            ",".join(df["Entry"]),
            "|".join(df["Name"]),
        ]
        print(gene, ":\n", pathway_map.loc[gene])
    pathway_map.to_excel(output_filename, sheet_name="AVENIO Gene <==> Pathway")


def assign_most_common_pathway(input_filename, output_filename):
    """
    From all pathways in which a gene is involved, select the most frequently occuring.
    """
    df = pd.read_excel(input_filename)
    df.set_index(df.columns[0], inplace=True)
    top_pathway = pd.DataFrame(index=df.index, columns=["pathway", "name"])

    # Count occurences of each pathway.
    pathway_counts = defaultdict(int)
    pathway_names = {}
    df["entries"] = df["entries"].apply(lambda x: x.split(","))
    df["names"] = df["names"].apply(lambda x: x.split("|"))
    for entries, names in zip(df["entries"], df["names"]):
        for pathway, name in zip(entries, names):
            pathway_names[pathway] = name
            pathway_counts[pathway] += 1

    def select_most_common_pathway(entries_column):
        entry_occurences = tuple(pathway_counts[e] for e in entries_column)
        i = np.argmax(entry_occurences)
        return entries_column[i]

    # For each gene, assign pathway with most genes.
    top_pathway["pathway"] = df["entries"].apply(select_most_common_pathway)
    top_pathway["name"] = top_pathway["pathway"].map(pathway_names)
    top_pathway.to_excel(output_filename)


fetch_pathways(output_filename="/tmp/gene_pathway_all.xlsx")
assign_most_common_pathway(
    input_filename="/tmp/gene_pathway_all.xlsx",
    output_filename="gene_pathway_most_frequent.xlsx",
)
