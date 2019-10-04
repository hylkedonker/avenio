import unittest

import pandas as pd

from transform import patient_allele_frequencies


class TestTransforms(unittest.TestCase):
    """
    Test utility functions in transform module.
    """

    def setUp(self):
        """
        Generate Avenio test data.
        """
        # Allele frequencies and corresponding column names.
        F_allele = [[0.1, 0.2], [0.3, 0.5]]
        allele_columns = ["T0: Allele \nFraction", "T1: Allele Fraction"]
        self.mutations = pd.DataFrame(F_allele, columns=allele_columns)
        # Number patients from 1 .. N.
        self.mutations["Patient ID"] = range(1, self.mutations.shape[0] + 1)
        # Generate name by going down the alphabet.
        self.mutations["Gene"] = (
            (ord("a") + self.mutations["Patient ID"]).astype(int).apply(chr)
        )
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
