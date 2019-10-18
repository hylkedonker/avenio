import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
from scipy.stats import pearsonr
from sklearn.tree import DecisionTreeClassifier

from transform import (
    ClassifierAsTransformer,
    get_top_correlated,
    patient_allele_frequencies,
)


class TestTransforms(unittest.TestCase):
    """
    Test utility functions in transform module.
    """

    def setUp(self):
        """
        Generate Avenio test data.
        """
        # Patient ids:
        patients = [1, 1, 2, 3, 3, 3, 3]

        # Allele frequencies and corresponding column names.
        F_allele = [
            # Patient 1.
            [0.1, 0.2],  # a
            [0.3, 0.5],  # b
            # Patient 2.
            [0.6, 0.4],  # c
            # Patient 3.
            [0.1, 0.05],  # a
            [0.5, 1.0],  # b
            # Duplicate mutations of gene "a" in patient 3.
            [0.01, 0.5],  # a
            [0.5, 0.01],  # a
        ]
        allele_columns = ["T0: Allele \nFraction", "T1: Allele Fraction"]
        self.mutations = pd.DataFrame(F_allele, columns=allele_columns)

        # Names of mutations.
        self.mutations["Gene"] = ["a", "b", "c", "a", "b", "a", "a"]
        self.mutations["Patient ID"] = patients

        # Vocabulary of genes, which we now simply take to be all the genes in
        # the data frame (i.e., vocabulary does not contain out of sample
        # genes).
        self.all_genes = self.mutations["Gene"].unique()

    def test_patient_allele_frequencies_errors(self):
        """
        Test error handling of `patient_allele_frequencies` function.
        """
        # Too many columns passed to `allele_freq_columns`.
        with self.assertRaises(ValueError):
            patient_allele_frequencies(
                self.mutations,
                gene_vocabulary=self.all_genes,
                allele_columns=["a", "b", "c"],
            )

        # Fail when passing non-existing colums.
        with self.assertRaises(KeyError):
            patient_allele_frequencies(
                self.mutations,
                gene_vocabulary=self.all_genes,
                allele_columns=["lorem", "ipsum"],
            )

        # Fail when some of the columns contain NA values.
        with self.assertRaises(ValueError):
            mutation_copy = self.mutations.copy()
            mutation_copy.iloc[0, 0] = None
            patient_allele_frequencies(
                mutation_copy, gene_vocabulary=self.all_genes
            )

    def test_patient_allele_frequencies_values(self):
        """
        Check that `patient_allele_frequencies` stores the correct values in the
        correct column.
        """
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

    def test_patient_allele_frequencies_duplicates(self):
        """
        Check if duplicates are correctly handled by
        `patient_allele_frequencies`.
        """
        # Check that first element is selected when handle_duplicates="ignore".
        df_freq = patient_allele_frequencies(
            self.mutations, self.all_genes, handle_duplicates="ignore"
        )
        self.assertEqual(df_freq.loc[3, "a"], -0.05)

        # Check that largest value is selected when handle_duplicates="max".
        df_freq = patient_allele_frequencies(
            self.mutations, self.all_genes, handle_duplicates="max"
        )
        self.assertEqual(df_freq.loc[3, "a"], 0.49)

        # Check that smallest value is selected when handle_duplicates="min".
        df_freq = patient_allele_frequencies(
            self.mutations, self.all_genes, handle_duplicates="min"
        )
        self.assertEqual(df_freq.loc[3, "a"], -0.49)

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

        # A single "c" mutation.
        self.assertTrue(np.all(top_df[c_mutation]["# gene 2"] == 1))
        # Four "a" mutations.
        self.assertTrue(np.all(top_df[a_mutation]["# gene 1"] == 4))

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
            top_count=2,
        )
        self.assertEqual(bottom_df.iloc[0]["correlation"], min_c)

        # 5) Check that DataFrame is correctly truncated.
        self.assertEqual(bottom_df.shape[0], 2)

        # 6) Generate top list with p-values. Check that p-values are correctly
        #    aligned.

        # Function to generate matrix with p values.
        def pearson_pval(x, y):
            return pearsonr(x, y)[1]

        pval_corr = df_freq.corr(method=pearson_pval).fillna(0)

        # Generate top list with p-values.
        top_df = get_top_correlated(correlations, pval_corr)
        gene_1, gene_2, p = top_df.loc[0, ["gene 1", "gene 2", "p-value"]]
        # Check that alignment is correct.
        self.assertEqual(pval_corr.loc[gene_1, gene_2], p)


class TestClassifierAsTransformer(unittest.TestCase):
    def setUp(self):
        """
        Initialise environment for testing.
        """
        self.seed = 1234
        np.random.seed(self.seed)

    def test_pipelines(self):
        """
        Verify that the transformer wrapper classifier works as expected.
        """
        X = np.random.random([10, 2])
        y = ["PD", "SD", "PR", "PD", "PR", "SD", "CR", "PD", "PR", "SD"]
        tree = DecisionTreeClassifier(random_state=self.seed).fit(X, y)
        tree_transformer = ClassifierAsTransformer(
            classifier=DecisionTreeClassifier(random_state=self.seed), encoder=None
        ).fit(X, y)
        np.testing.assert_array_equal(
            tree.predict(X), tree_transformer.transform(X)
        )
