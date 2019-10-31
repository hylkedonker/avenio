import unittest

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from models import ClassifierAsTransformer, UniqueFeatureFilter


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
