from collections import defaultdict
from functools import wraps
from typing import Callable

import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    KFold,
    StratifiedKFold,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
)
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


def _double_cross_validate(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    m_inner: int = 5,
    n_outer: int = 5,
    positive_class=1,
    verbose=True,
):
    """
    Group predictions and ground truth values, per fold.
    """

    seed = 1234
    if m_inner > 1:
        inner_loop = StratifiedKFold(n_splits=m_inner, shuffle=True, random_state=seed)
        clf = GridSearchCV(
            estimator=pipeline,
            param_grid=get_hyper_param_grid(pipeline),
            scoring="roc_auc",
            cv=inner_loop,
            n_jobs=-1,
            verbose=3 * int(verbose),
        )
    outer_loop = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    y_pred = []
    y_true = []
    for (train, test) in outer_loop.split(X, y):
        X_train, y_train = X.iloc[train], y.iloc[train]
        if m_inner > 1:
            estimator = clf.fit(X_train, y_train).best_estimator_
        else:
            estimator = pipeline.fit(X_train, y_train)
        class_index = list(estimator.classes_).index(positive_class)

        X_test, y_test = X.iloc[test], y.iloc[test]
        y_prob = estimator.predict_proba(X_test)[:, class_index]
        y_pred.append(y_prob)
        y_true.append(y_test)

    return y_true, y_pred


def double_cross_validate_average(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    metrics: dict,
    m_inner: int = 5,
    n_outer: int = 5,
    positive_class=1,
    verbose=True,
) -> tuple:
    """
    Calculate metrics by averaging cross-validation predictions.
    """
    y_true, y_pred = _double_cross_validate(
        pipeline, X, y, m_inner, n_outer, positive_class, verbose
    )
    metric_values = defaultdict(list)
    for name, function in metrics.items():
        for i in range(n_outer):
            value = function(y_true[i], y_pred[i])
            metric_values[name].append(value)

    return (
        {k: np.mean(v, axis=0) for k, v in metric_values.items()},
        {k: np.std(v, axis=0) for k, v in metric_values.items()},
    )


def double_cross_validate_concatenate(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    metrics: dict,
    m_inner: int = 5,
    n_outer: int = 5,
    positive_class=1,
    verbose=True,
):
    """
    Calculate metrics by concatenating cross-validation predictions.

    Apples-to-Apples in Cross-Validation Studies:
        Pitfalls in Classifier Performance Measurement
    """
    y_true, y_pred = _double_cross_validate(
        pipeline, X, y, m_inner, n_outer, positive_class, verbose
    )
    metric_values = {}
    for name, function in metrics.items():
        # Unravel arrays to flat vector.
        y_true_flat = np.concatenate(y_true)
        y_pred_flat = np.concatenate(y_pred)
        metric_values[name] = function(y_true_flat, y_pred_flat)

    return metric_values


def metrics_report(
    clf,
    X,
    y,
    positive_class,
    operating_point: float = 0.5,
    how="average",
    random_state=1234,
):
    """
    Compute average metrics using cross-validation.
    """
    mean_fpr = np.linspace(0, 1, 100)

    def tpr(y_true, y_pred):
        fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=positive_class)
        if how == "average":
            # Calculate interpolated true positive rate for ROC curve.
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            return interp_tpr
        return tpr

    def fpr(y_true, y_pred):
        if how == "average":
            # FPRs as a grid.
            return mean_fpr
        fpr, _, _ = roc_curve(y_true, y_pred, pos_label=positive_class)
        return fpr

    def precisions(y_true, y_pred):
        precision, _, _ = precision_recall_curve(
            y_true, y_pred, pos_label=positive_class
        )
        return precision

    def recalls(y_true, y_pred):
        _, recalls, _ = precision_recall_curve(y_true, y_pred, pos_label=positive_class)
        return recalls

    metrics = {
        "tprs": tpr,
        "fprs": fpr,
        "roc_auc": roc_auc_score,
        "accuracy": lambda y_true, y_pred: accuracy_score(
            y_true, y_pred > operating_point
        ),
        "confusion_matrix": lambda y_true, y_pred: confusion_matrix(
            y_true, y_pred > operating_point, normalize="all"
        ),
        "average_precision": average_precision_score,
        "calibration": calibration_curve,
        "f_1_score": lambda y_true, y_pred: f1_score(
            y_true, y_pred > operating_point, pos_label=positive_class
        ),
    }
    if how == "average":
        return double_cross_validate_average(
            clf, X, y, metrics, positive_class=positive_class
        )
    elif how == "concatenate":
        metrics["precision"] = precisions
        metrics["recall"] = recalls
        return double_cross_validate_concatenate(
            clf, X, y, metrics, positive_class=positive_class
        )
    raise ValueError(f"Wrong value `how`={how}.")


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
                    scoring="roc_auc",
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


def data_frame_to_disk(
    X: pd.DataFrame,
    all_filename: str,
    train_filename: str,
    test_filename: str,
    random_state: int = 1234,
):
    """
    Write data to disk.
    """
    f_test = 0.2
    X.sort_index(inplace=True)
    X_train, X_test = train_test_split(X, test_size=f_test, random_state=random_state)
    X.to_csv(all_filename + ".tsv", sep="\t")
    X.to_excel(all_filename + ".xlsx")
    X_train.to_csv(train_filename + ".tsv", sep="\t")
    X_test.to_csv(test_filename + ".tsv", sep="\t")
