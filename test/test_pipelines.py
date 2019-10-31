import unittest

import pandas as pd
import numpy as np

from pipelines import select_phenotype_columns, select_no_phenotype_columns


class TestPipelineUtils(unittest.TestCase):
    def test_filter_functions(self):
        """
        Test the phenotype filtering functions.
        """
        X = pd.DataFrame(
            # Generate 4x4 matrix.
            np.array(range(4 ** 2)).reshape(4, 4),
            columns=["a", "therapyline", "c", "lungmeta"],
        )
        # Test that `select_phenotype_columns` filters out all other columns.
        np.testing.assert_array_equal(
            select_phenotype_columns(X), X[["therapyline", "lungmeta"]]
        )
        # And test that `select_no_phenotype_columns` does precisely the
        # opposite.
        np.testing.assert_array_equal(
            select_no_phenotype_columns(X), X[["a", "c"]]
        )
