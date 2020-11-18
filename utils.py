from collections import defaultdict
from functools import wraps
from typing import Callable, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from models import get_hyper_param_grid


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
