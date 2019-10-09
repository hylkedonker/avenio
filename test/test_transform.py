import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from transform import get_top_correlated, patient_allele_frequencies


class TestTransforms(unittest.TestCase):
    """
    Test utility functions in transform module.
    """

    def setUp(self):
        """
        Generate Avenio test data.
        """
        # Patient ids:
        patients = [1, 1, 2, 3, 3]

        # Allele frequencies and corresponding column names.
        F_allele = [
            # Patient 1.
            [0.1, 0.2],
            [0.3, 0.5],
            # Patient 2.
            [0.6, 0.4],
            # Patient 3.
            [0.1, 0.05],
            [0.5, 1.0],
        ]
        allele_columns = ["T0: Allele \nFraction", "T1: Allele Fraction"]
        self.mutations = pd.DataFrame(F_allele, columns=allele_columns)

        # Names of mutations.
        self.mutations["Gene"] = ["a", "b", "c", "a", "b"]
        self.mutations["Patient ID"] = patients

        # Vocabulary of genes, which we now simply take to be all the genes in
        # the data frame (i.e., vocabulary does not contain out of sample
        # genes).
        self.all_genes = self.mutations["Gene"].unique()

    def test_patient_allele_frequencies(self):
        """
        Validate correct of `patient_allele_frequencies` function.
        """
        # Two many columns passed to `allele_freq_columns`.
        with self.assertRaises(ValueError):
            patient_allele_frequencies(
                self.mutations,
                gene_vocabulary=self.all_genes,
                allele_freq_columns=["a", "b", "c"],
            )

        # Fail when passing non-existing colums.
        with self.assertRaises(KeyError):
            patient_allele_frequencies(
                self.mutations,
                gene_vocabulary=self.all_genes,
                allele_freq_columns=["lorem", "ipsum"],
            )

        # Fail when some of the columns contain NA values.
        with self.assertRaises(ValueError):
            mutation_copy = self.mutations.copy()
            mutation_copy.iloc[0, 0] = None
            patient_allele_frequencies(
                mutation_copy, gene_vocabulary=self.all_genes
            )

        df_freq = patient_allele_frequencies(self.mutations, self.all_genes)

        reference_mutation_names = ["a", "b", "c"]
        # First patient has two mutations.
        assert_series_equal(
            df_freq.loc[1],
            pd.Series([0.1, 0.2, 0.0], index=reference_mutation_names),
            check_names=False,
        )
        # Second patient has single mutation.
        assert_series_equal(
            df_freq.loc[2],
            pd.Series([0.0, 0.0, -0.2], index=reference_mutation_names),
            check_names=False,
        )

    def test_top_correlated(self):
        """
        Test the `get_top_correlated` function.
        """
        # First calculate frequencies and corresponding correlations.
        df_freq = patient_allele_frequencies(self.mutations, self.all_genes)
        correlations = df_freq.corr()
        top_df = get_top_correlated(
            correlations,
            gene_counts=self.mutations["Gene"].value_counts(),
            ascending=False,
        )

        # 1) Check that there are no diagonal elements.
        self.assertFalse(np.any(top_df["gene 1"] == top_df["gene 2"]))

        # 2) Check that there are no duplicate permutations of the gene
        # correlations [since (i, j) ~ (j, i)].
        i, j = top_df[["gene 1", "gene 2"]].iloc[0]
        permuted = (top_df["gene 1"] == j) & (top_df["gene 2"] == i)
        # We can not find a permutation of the first record.
        self.assertEqual(len(top_df[permuted]), 0)

        # 3) Check alignment mutation occurences.
        a_mutation = top_df["gene 1"] == "a"
        c_mutation = top_df["gene 2"] == "c"
        # Mutation c occurs 1 time, mutation a occurs 2 times.
        self.assertTrue(np.all(top_df[c_mutation]["# gene 2"] == 1))
        self.assertTrue(np.all(top_df[a_mutation]["# gene 1"] == 2))

        # 4) Check descending and ascending order.
        # First element is largest value.
        flat_correlations = (
            correlations.values - np.identity(correlations.shape[0])
        ).flatten()

        min_c, max_c = min(flat_correlations), max(flat_correlations)
        self.assertEqual(top_df.iloc[0]["correlation"], max_c)
        bottom_df = get_top_correlated(
            correlations,
            gene_counts=self.mutations["Gene"].value_counts(),
            ascending=True,
        )
        self.assertEqual(bottom_df.iloc[0]["correlation"], min_c)
