import os
import tempfile
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
from scipy.stats import pearsonr

from source import categorical_columns_to_lower, read_preprocessed_data

from transform import (
    clean_mutation_columns,
    load_process_and_store_spreadsheets,
    get_top_correlated,
    transform_column_pair,
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
        patients = [1, 1, 2, 3, 3, 3, 3, 4, 4]

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
            # Patient 4, with ND value.
            [0.01, "ND"],  # a
            ["2%", "3%"],  # b
        ]
        self.allele_columns = ["T0: Allele \nFraction", "T1: Allele Fraction"]
        self.raw_mutations = pd.DataFrame(F_allele, columns=self.allele_columns)

        # Names of mutations.
        self.raw_mutations["Gene"] = ["a", "b", "c", "a", "b", "a", "a", "a", "b"]
        self.raw_mutations["Patient ID"] = patients
        self.mutations, self.dirty_mutations = clean_mutation_columns(
            self.raw_mutations, columns_to_number=self.allele_columns
        )

        # Vocabulary of genes, which we now simply take to be all the genes in
        # the data frame (i.e., vocabulary does not contain out of sample
        # genes).
        self.all_genes = self.raw_mutations["Gene"].unique()

    def test_patient_allele_frequencies_errors(self):
        """
        Test error handling of `patient_allele_frequencies` function.
        """
        # Too many columns passed to `allele_freq_columns`.
        with self.assertRaises(ValueError):
            transform_column_pair(
                self.mutations,
                gene_vocabulary=self.all_genes,
                column_pair=["a", "b", "c"],
            )

        # Fail when passing non-existing colums.
        with self.assertRaises(KeyError):
            transform_column_pair(
                self.mutations,
                gene_vocabulary=self.all_genes,
                column_pair=["lorem", "ipsum"],
            )

        # Fail when some of the columns contain NA values.
        with self.assertRaises(ValueError):
            mutation_copy = self.mutations.copy()
            mutation_copy.iloc[0, 0] = None
            transform_column_pair(mutation_copy, gene_vocabulary=self.all_genes)

    def test_patient_allele_frequencies_values(self):
        """
        Check that `patient_allele_frequencies` stores the correct values in the
        correct column.
        """
        df_freq = transform_column_pair(self.mutations, self.all_genes)

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
        df_freq = transform_column_pair(
            self.mutations, self.all_genes, handle_duplicates="ignore"
        )
        self.assertEqual(df_freq.loc[3, "a"], -0.05)

        # Check that largest value is selected when handle_duplicates="max".
        df_freq = transform_column_pair(
            self.mutations, self.all_genes, handle_duplicates="max"
        )
        self.assertEqual(df_freq.loc[3, "a"], 0.49)

        # Check that smallest value is selected when handle_duplicates="min".
        df_freq = transform_column_pair(
            self.mutations, self.all_genes, handle_duplicates="min"
        )
        self.assertEqual(df_freq.loc[3, "a"], -0.49)

    def test_top_correlated(self):
        """
        Test the `get_top_correlated` function.
        """
        # First calculate frequencies and corresponding correlations.
        df_freq = transform_column_pair(self.mutations, self.all_genes)
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
        # Five "a" mutations.
        self.assertTrue(np.all(top_df[a_mutation]["# gene 1"] == 5))

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

    def test_categorical_columns_to_lower(self):
        """
        Test that categorical columns are converted to lower case.
        """
        df = pd.DataFrame(np.array([range(4), ("a", "b", "A", "c")]).T)
        df2 = categorical_columns_to_lower(df)

        # Check that first column remains untouched.
        pd.testing.assert_series_equal(df.iloc[:, 0], df2.iloc[:, 0])

        # And check that elements have been turned in to lower case.
        self.assertTrue(df2.iloc[:, 1].equals(pd.Series(["a", "b", "a", "c"])))

    def _generate_train_test_filenames(self):
        """
        Generate filenames for the `load_process_and_store_spreadsheets` function.
        """
        # Generate set of filenames.
        d = tempfile.mkdtemp()
        return {
            "all_filename": os.path.join(d, "all_data.tsv"),
            "train_filename": os.path.join(d, "train.tsv"),
            "test_filename": os.path.join(d, "test.tsv"),
        }

    def test_clean_mutation_columns(self):
        """
        Test that columns are correctly cleaned.
        """
        clean_df, dirty_df = clean_mutation_columns(
            self.raw_mutations, columns_to_number=self.allele_columns
        )
        # The column containing 'ND' should have been replaced by 0.0. This means that
        # `dirty_df` is empty.
        self.assertEqual(dirty_df.shape[0], 0)
        percentage_mutation = (clean_df["Patient ID"] == 4) & (clean_df["Gene"] == "b")

        # Check that percentages have been converted to numbers.
        self.assertTrue(
            np.all(
                clean_df.loc[percentage_mutation, self.allele_columns] == [0.02, 0.03]
            )
        )

    def test_load_process_and_store_spreadsheets(self):
        """
        Perform consistency checks on the loaded data.
        """
        # Generate set of filenames.
        filename_kwargs = self._generate_train_test_filenames()

        # Process all the data and store to disk.
        load_process_and_store_spreadsheets(**filename_kwargs)

        # Reload the stored data.
        X_all, y_all = read_preprocessed_data(filename_kwargs["all_filename"])
        X_train, y_train = read_preprocessed_data(filename_kwargs["train_filename"])
        X_test, y_test = read_preprocessed_data(filename_kwargs["test_filename"])

        # Test that the total row counts add up.
        self.assertEqual(X_all.shape[0], X_train.shape[0] + X_test.shape[0])

        # The columns must be identical.
        self.assertTrue(np.all(X_all.columns == X_train.columns))
        self.assertTrue(np.all(X_train.columns == X_test.columns))

        def harmonic_mean(t0, t1):
            return t0 * t1 / (t0 + t1)

        # Test that the phenotype data remains unchanged when using different mutation
        # data spreadsheets.

        # 1. first read the default mutation spreadsheet.
        default_filename_kwargs = self._generate_train_test_filenames()
        load_process_and_store_spreadsheets(
            spread_sheet_filename="2019-08-27_PLASMA_DEFAULT_Results_Groningen.xlsx",
            transformation=harmonic_mean,
            **default_filename_kwargs
        )
        X_default, y_default = read_preprocessed_data(
            default_filename_kwargs["all_filename"]
        )

        # 2. read the somatic mutation spreadsheet.
        somatic_filename_kwargs = self._generate_train_test_filenames()
        load_process_and_store_spreadsheets(
            spread_sheet_filename="2019-08-27_PLASMA_SOMATIC_Results_Groningen.xlsx",
            transformation=harmonic_mean,
            **somatic_filename_kwargs
        )
        X_somatic, y_somatic = read_preprocessed_data(
            somatic_filename_kwargs["all_filename"]
        )
        # 3. Check that the X and y dataframes are aligned.
        np.all(X_somatic.index == y_somatic.index)
        np.all(X_default.index == y_default.index)

        # 4. Check that the phenotype data is the same in both cases.
        patients_in_common = list(
            set(X_default.index).intersection(set(X_somatic.index))
        )
        from source import phenotype_features

        df_default = X_default.loc[patients_in_common, phenotype_features]
        df_somatic = X_somatic.loc[patients_in_common, phenotype_features]
        pd.testing.assert_frame_equal(df_default, df_somatic)

        # self.assertTrue(set(X_default.index).issubset(set(X_somatic.index)))

        # pd.testing.assert_series_equal(
        #     X_default["gender"].sort_index(), X_somatic["gender"].sort_index()
        # )
        # pd.testing.assert_frame_equal(y_somatic.sort_index(), y_default.sort_index())
