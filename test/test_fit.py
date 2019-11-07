import unittest

import numpy as np
import pandas as pd

from fit import fit_categorical_survival


class TestExponentialDecay(unittest.TestCase):
    def test_fit_categorical_survival(self):
        # Generate exponential data for the tests.
        tau_1 = 50
        tau_2 = 55
        t = np.linspace(0, 100, 100)
        y_1 = np.exp(-t / tau_1 * np.log(2))
        x_1 = ["cat 1"] * len(y_1)
        y_2 = np.exp(-t / tau_2 * np.log(2))
        x_2 = ["cat 2"] * len(y_2)

        df = pd.DataFrame({"x": x_1 + x_2, "y": y_1 + y_2})
        import ipdb
        ipdb.set_trace()
