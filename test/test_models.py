import unittest

import numpy as np
import pandas as pd

from models import UniqueFeatureFilter


class TestUniqueFeatureFilter(unittest.TestCase):
    def test_filter(self):
        """
        Test thessholding feature filter.
        """
        X = pd.DataFrame(
            np.array([range(4), [0] * 3 + [1], [0] * 2 + [1, 2]]).T,
            columns=["a", "b", "c"],
        )
        f = UniqueFeatureFilter(thresshold=2)
        f.fit(X)
        # Check that correct columns are filtered.
        self.assertEqual(set(f.columns_to_keep), {"a", "c"})
        # Test that array is correctly transformed.
        np.testing.assert_array_equal(f.transform(X), X[["a", "c"]])
