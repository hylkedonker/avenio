import numpy as np
import scipy as sp

from transform import survival_histograms


def fit_half_life(t, p):
    """
    Calculate half life (and r-value) from exponentially suppressed distribution.
    """
    slope, intercept, r_value, p_value, std_err = sp.stats.linregress(t, np.log(p))
    tau = -1.0 / slope * np.log(2)
    return (tau, r_value)


def fit_categorical_survival(x, y):

    # Histogram and cumulative histogram of survival data.
    (t, p), (t_cum, p_cum) = survival_histograms(x, y)
    tau, r = fit_half_life(t_cum, p_cum)
