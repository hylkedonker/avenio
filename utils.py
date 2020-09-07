from collections import defaultdict
from functools import wraps
from typing import Callable, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from const import get_hyper_param_grid


def get_categorical_columns(
    data_frame: pd.DataFrame, uniqueness_thresshold: Optional[float] = None
) -> list:
    """
    Find all non-numeric columns.

    Args:
        data_frame (pd.DataFrame): Analyse columns from this data frame.
        uniqueness_thresshold (float): If less than this fraction of the values are
            unique, than consider the column categorical.
    """
    categorical_columns = []
    for column in data_frame.columns:
        values = data_frame[column]

        if values.dtype.name == "category":
            categorical_columns.append(column)
            continue

        # This is a dirty way to check if it is non-numeric, but pandas thinks
        # all the columns are strings.
        try:
            float(values.iloc[0])
        except ValueError:
            categorical_columns.append(column)
            continue

        # If it is numeric, but lots of non-zero values are identical, consider it
        # categorical.
        if uniqueness_thresshold is not None:
            # Correct for sparseness, by ignoring zero values.
            if 0 in values.unique() and values.nunique() > 1:
                non_sparse_counts = len(values) - values.value_counts()[0]
                if (values.nunique() - 1) / non_sparse_counts <= uniqueness_thresshold:
                    categorical_columns.append(column)
            elif values.nunique() / len(values) <= uniqueness_thresshold:
                categorical_columns.append(column)

    return categorical_columns


def get_numerical_columns(
    data_frame: pd.DataFrame,
    ignore_columns: list = [],
    uniqueness_thresshold: Optional[float] = None,
) -> list:
    """
    Single out numerical columns.

    Args:
        ignore_columns (list): Remove these columns from the consideration.
        uniqueness_thresshold (float): If more than this fraction of the values are
            unique, consider the column numerical.
    """
    categorical_columns = get_categorical_columns(data_frame, uniqueness_thresshold)

    def is_numeric_and_not_ignored(column):
        """ Columns not categorical are numeric. """
        if column not in categorical_columns and column not in ignore_columns:
            return True
        return False

    numerical_columns = list(filter(is_numeric_and_not_ignored, data_frame.columns))
    return numerical_columns


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
            if y.dtype == np.object_:
                splits = StratifiedKFold(n_splits=k).split(X, y)
            else:
                splits = KFold(n_splits=k).split(X)
            for train, test in splits:
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


def double_cross_validate(m_inner: int = 5, n_outer: int = 5, verbose=True) -> Callable:
    """
    Perform `m_inner` x `n_outer` double (or netsed) cross validation.
    """

    def wrap_m_x_n_fold(function: Callable) -> Callable:
        """
        Wrapper for `function`.
        """

        @wraps(function)
        def cross_validate_m_x_n(
            pipeline, X: pd.DataFrame, y: pd.DataFrame, *args, **kwargs
        ):
            seed = 1234
            if m_inner > 1:
                inner_loop = StratifiedKFold(
                    n_splits=m_inner, shuffle=True, random_state=seed
                )
                clf = GridSearchCV(
                    estimator=pipeline,
                    param_grid=get_hyper_param_grid(pipeline),
                    cv=inner_loop,
                    n_jobs=-1,
                    verbose=3 * int(verbose),
                )
            outer_loop = StratifiedKFold(
                n_splits=n_outer, shuffle=True, random_state=seed
            )
            outputs = defaultdict(list)
            for i, (train, test) in enumerate(outer_loop.split(X, y)):
                X_train, y_train = X.iloc[train], y.iloc[train]
                if m_inner > 1:
                    estimator = clf.fit(X_train, y_train).best_estimator_
                else:
                    estimator = pipeline.fit(X_train, y_train)

                X_test, y_test = X.iloc[test], y.iloc[test]
                predictions = function(estimator, X_test, y_test, *args, **kwargs)
                if isinstance(predictions, dict):
                    for k, v in predictions.items():
                        outputs[k].append(v)
                else:
                    outputs["metric"].append(v)

            if len(outputs.keys()) == 1:
                k = next(iter(outputs.keys()))
                return np.mean(outputs[k], axis=0), np.std(outputs[k], axis=0)
            return (
                {k: np.mean(v, axis=0) for k, v in outputs.items()},
                {k: np.std(v, axis=0) for k, v in outputs.items()},
            )

        return cross_validate_m_x_n

    return wrap_m_x_n_fold


def get_sub_pipeline(pipeline, step: int):
    """
    Get part of the pipeline upto and including `step`.
    """
    return Pipeline(pipeline.steps[:step])
