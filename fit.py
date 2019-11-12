from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp

from transform import survival_histograms


def fit_half_life(t, p):
    """
    Calculate half life (and r-value) from exponentially suppressed distribution.
    """
    slope, intercept, r_value, p_value, std_err = sp.stats.linregress(t, np.log(p))
    tau = -1.0 / slope * np.log(2)
    return (tau, r_value)


def fit_categorical_survival(
    x: pd.Series, y: pd.Series, plot: str = "none"
) -> pd.DataFrame:
    """
    Fit exponentially decaying function for each category.
    """
    # Parse argument.
    if plot is not None and plot.lower() == "none":
        plot = None

    # Store result in DataFrame.
    columns = ["tau", "sigma_t", "n", "r"]
    df = pd.DataFrame(index=list(x.unique()).append("all"), columns=columns)

    # 1) First do a fit for the data combined.
    # Calculate a histogram and cumulative histogram of the survival data.
    _, (t_cum, p_cum) = survival_histograms(y)
    tau, r = fit_half_life(t_cum, p_cum)
    df.loc["all", columns] = tau, np.std(y), len(y), r

    # 2) Repeat, but now for every category.
    for category in x.unique():
        # Filter out all records for given cateogry.
        y_category = y[x == category]

        # Histogram and cumulative histogram of survival data.
        (t, p), (t_cum, p_cum) = survival_histograms(y_category)
        tau, r = fit_half_life(t_cum, p_cum)

        df.loc[category, columns] = tau, np.std(y_category), len(y_category), r
        if plot:
            if plot == "pdf":
                plt.plot(t, p, label=category)
            if plot == "cdf":
                plt.plot(t_cum, p_cum, label=category)

    if plot:
        plt.legend(frameon=False)
        plt.tight_layout()

    return df


def categorical_signal(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the signal for given categories.

    See unit test `test_signal` for specific constraints that apply to the data frame
    `X`.
    """
    combined_fit = X.loc["all"]
    X = X.drop("all")
    a = {}
    for i, row_i in enumerate(X.index):
        for row_j in X.index[i + 1 :]:
            # Amount of signal.
            s = np.abs(X.loc[row_i, "tau"] - X.loc[row_j, "tau"])
            a[f"{row_i}-{row_j}"] = {
                "delta tau": s,
                "signal to noise": s / combined_fit["sigma_t"],
            }
    return pd.DataFrame(a).T
