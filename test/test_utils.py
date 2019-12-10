import unittest

import numpy as np

from utils import bootstrap


class TestUtils(unittest.TestCase):
    def test_bootstrap_decorator(self):
        """
        Test the function of the bootstrap decorator.
        """
        X = np.arange(0, 9)
        y = np.ones(9)

        # Three fold bootstrapping of a simple average function.
        @bootstrap(k=3)
        def average(X_train, y_train, X_test=None, y_test=None):
            """ Ignore everything but the training data. """
            return np.mean(X_train)

        # Values of each of the folds.
        ground_truth = [2.5, 5.5, 4.0]

        np.testing.assert_array_equal(
            average(X, y), np.array([np.mean(ground_truth), np.std(ground_truth)])
        )
