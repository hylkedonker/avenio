import numpy as np
import pandas as pd

from sklearn.model_selection import KFold


def get_categorical_columns(data_frame: pd.DataFrame) -> list:
    """
    Convert all text columns to lower case.
    """
    categorical_columns = []
    for column in data_frame.columns:
        # This is a dirty way to check if it is non-numeric, but pandas thinks
        # all the columns are strings.
        try:
            float(data_frame[column].iloc[0])
        except ValueError:
            categorical_columns.append(column)

    return categorical_columns


def bootstrap(k):
    """
    Decorator for bootstrapping function `k` times.

    Signature of the function to decorate:
        f(X_train, y_train, X_test, y_test, *args, **kwargs) -> np.ndarray
    Signature of returned function:
        f(X, y, *args, **kwargs) -> np.ndarray
    """

    def wrap_k_fold(function):
        """
        Wrapper function for `k` fold bootstrapping.
        """

        def bootstrap_k_fold(X, y, *args, **kwargs):
            """
            Carry out the actual bootstrapping.
            """
            bootstrapped_results = []

            # k-fold cross validation of training size dependence.
            # Keep track of scores for this particular fold.
            for train, test in KFold(n_splits=k).split(X):
                if isinstance(X, pd.DataFrame):
                    X_train, X_test, y_train, y_test = (
                        X.iloc[train],
                        X.iloc[test],
                        y.iloc[train],
                        y.iloc[test],
                    )
                else:
                    X_train, X_test, y_train, y_test = (
                        X[train],
                        X[test],
                        y[train],
                        y[test],
                    )

                # Calculate the actual function.
                values = function(X_train, y_train, X_test, y_test, *args, **kwargs)

                bootstrapped_results.append(values)

            bootstrapped_results = np.array(bootstrapped_results)

            return (
                np.mean(bootstrapped_results, axis=0),
                np.std(bootstrapped_results, axis=0),
            )

        return bootstrap_k_fold

    return wrap_k_fold
