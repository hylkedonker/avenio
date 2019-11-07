import unittest

import numpy as np
import pandas as pd

from fit import fit_categorical_survival, categorical_signal


class TestExponentialDecay(unittest.TestCase):
    def setUp(self):
        # Make sure that tests are reproducible.
        RANDOM_SEED = 1234
        np.random.seed(RANDOM_SEED)

    def test_fit_categorical_survival(self):
        # Generate exponential data for the tests.
        tau_1 = 10
        tau_2 = 12
        t = np.linspace(0, 100, 50)
        y_1 = np.exp(-t / tau_1 * np.log(2))
        x_1 = ["cat 1"] * len(y_1)
        y_2 = np.exp(-t / tau_2 * np.log(2))
        x_2 = ["cat 2"] * len(y_2)

        # Combine the two data sets.
        x = np.concatenate([x_1, x_2])
        y = np.concatenate([y_1, y_2])
        # And shuffle them.
        df = pd.DataFrame({"x": x, "y": y}).sample(frac=1)

        statistics = fit_categorical_survival(df["x"], df["y"], plot=None)

    def test_signal(self):
        """
        Check that the signal is correctly calculated.
        """
        # Generate some test data that could have come from the
        # `fit_categorical_survival` function.
        X = np.array(
            [
                [216.83738685734016, 305.87701957486115, -0.9500320032827645],
                [329.65693981123405, 340.06219039071857, -0.9979758494324196],
                [202.73547762442317, 253.24880192938792, -0.9742255012638429],
                [217.3016880451117, 203.3360544035616, -0.9164690475648419],
                [303.42238460145785, 152.4624543945164, -0.9605756635760584],
            ]
        )
        fit_statistics = pd.DataFrame(
            X,
            columns=["tau", "sigma_t", "r"],
            index=["all", "smoker", "previous", "unknown", "non smoker"],
        )

        s = categorical_signal(fit_statistics)

        self.assertEqual(s.shape[0], 6)
        np.testing.assert_almost_equal(
            s.loc["smoker-unknown", "signal to noise"], 0.36732165078071266
        )
        np.testing.assert_almost_equal(
            s.loc["smoker-unknown", "signal"], 112.35525176612236
        )
