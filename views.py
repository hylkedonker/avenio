from copy import copy
from typing import Iterable

import graphviz
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from fit import categorical_signal, fit_categorical_survival
from models import SparseFeatureFilter
from pipelines import (
    calculate_pass_through_column_names_Richard,
    reconstruct_categorical_variable_names_Richard,
)

matplotlib.rc("font", size=22)
matplotlib.rc("lines", linewidth=4)
matplotlib.rc("figure", autolayout=True)
matplotlib.rc("ytick", labelsize="large")
matplotlib.rc("xtick", labelsize="large")
matplotlib.rc("axes", labelsize="xx-large", titlesize="xx-large")
matplotlib.rc("legend", fontsize="x-large")


def plot_confusion_matrix(
    y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    return ax


def remove_classifier_from_pipelines(pipelines: dict) -> dict:
    """
    Make new pipelines with last step (= classifier) removed.
    """
    truncated_pipelines = {}
    for name, p in pipelines.items():
        # Ignore classifiers.
        if not isinstance(p, Pipeline) or len(p.steps) == 1:
            continue
        # Make a copy, and delete last step.
        p_copy = copy(p)
        del p_copy.steps[-1]
        truncated_pipelines[name] = p_copy

    return truncated_pipelines


def view_pipelines(pipelines: dict, X, y, random_state: int = 1234):
    """
    Generate lower dimensional projection of transformed data.
    """
    truncated_pipelines = remove_classifier_from_pipelines(pipelines)
    for name, pipeline in truncated_pipelines.items():
        plt.figure()
        plt.title(name)
        X_T = pipeline.fit_transform(X)
        # Project onto 2D.
        X_subspace = TSNE(random_state=random_state).fit_transform(X_T)
        sns.scatterplot(x=X_subspace[:, 0], y=X_subspace[:, 1], hue=y)


def filter_outliers(array: Iterable, outlier_indices: list) -> np.array:
    return np.array([a for i, a in enumerate(array) if i not in outlier_indices])


def view_as_exponential(
    t, p, outlier_indices=[], markers=["o", "-"], text_location="above", label=""
):
    # Make a fit without outliers.
    slope, intercept, r_value, p_value, std_err = sp.stats.linregress(
        filter_outliers(t, outlier_indices), filter_outliers(np.log(p), outlier_indices)
    )

    tau = -1.0 / slope * np.log(2)

    tau_text = r"$\tau={:.0f}$ days".format(tau)
    R_text = "$R^2={:.2f}$".format(r_value ** 2)
    fit_text = tau_text + ", " + R_text

    p = plt.plot(t, np.log(p), markers[0], label=label)
    ca = plt.gca()
    xlim = np.array(ca.get_xlim())
    plt.plot(
        xlim,
        slope * xlim + intercept,
        markers[1],
        label=fit_text,
        color=p[0].get_color(),
    )
    # plt.xlim([-1, max(t) + 1])
    plt.xlabel(r"$t$ (days)")
    plt.ylabel(r"$\ln[n(t)]$")

    # x_centre = (xlim[1] - xlim[0]) / 2.0
    # y_centre = x_centre * slope + intercept
    # text_loc = np.array([x_centre, y_centre])
    # if text_location == "above":
    #     text_loc *= 1.2
    # elif text_location == "below":
    #     text_loc * 0.8
    # else:
    #     text_loc = text_location

    # ca.text(
    #     text_loc[0],
    #     text_loc[1],
    #     fit_text,
    #     ha="center",
    #     va="center",
    #     # transform=ca.transAxes,
    #     fontsize="xx-large",
    # )
    # Location 3 is lower left corner.
    plt.legend(frameon=False, loc=3)


def categorical_signal_summary(
    X: pd.DataFrame, y: pd.Series, categorical_columns: list
) -> pd.DataFrame:
    """
    Make a summary of all categorical effects.
    """
    summary = pd.DataFrame()

    for category in categorical_columns:
        # Calculate survival statistics for given prior information.
        df = fit_categorical_survival(X[category], y)
        print(df)
        # Calculate signal.
        s = categorical_signal(df)
        # Add results to summary.
        s["item"] = s.index
        s["category"] = category
        # Add results to summary.
        summary = summary.append(s, ignore_index=True)

    return summary.set_index(["category", "item"])


def remove_parallel_coefficients(coefficients, names):
    """
    Remove coefficients that are (almost) equal but opposite in sign.
    """
    name_new, coef_new = [], []
    for i, y_i in enumerate(coefficients):
        if i < len(names) - 1:
            # If almost opposite in sign.
            if np.absolute(y_i + coefficients[i + 1]) < 0.01 * np.std(coefficients):
                continue

        name_new.append(names[i])
        coef_new.append(y_i)

    return coef_new, name_new


def view_linear_model_richard(pipeline):
    """
    Plot the coefficients of Richard model.
    """
    richard_classifier = pipeline.steps[-1][1]
    variable_names = reconstruct_categorical_variable_names_Richard(pipeline)
    # Concatenate with unaltered phenotype columns.
    variable_names.extend(calculate_pass_through_column_names_Richard())
    coefficients, names = remove_parallel_coefficients(
        richard_classifier.coef_, variable_names
    )

    with sns.plotting_context(font_scale=1.5):
        plt.figure(figsize=(8, 6))
        plt.xlabel(r"Slope")
        sns.barplot(x=coefficients, y=names, label="large")
        plt.tight_layout()


def view_linear_model_julian(p_julian):
    """
    Plot the coefficients of Richard model.
    """
    classifier = p_julian.steps[-1][1]
    column_names = p_julian.steps[-2][1].columns_to_keep_
    print(classifier.coef_)
    with sns.plotting_context(font_scale=1.5):
        plt.xlabel(r"Slope")
        sns.barplot(x=classifier.coef_, y=column_names, label="large")
        plt.tight_layout()


def view_decision_tree_julian(pipeline):
    """
    Plot the decision tree of a Julian pipeline.
    """
    tree_classifier = pipeline.steps[-1][1]
    # Consistency check of the Julian pipeline.
    assert isinstance(pipeline.steps[-2][1], SparseFeatureFilter)

    kwargs = {
        "out_file": None,
        "filled": True,
        "rounded": True,
        "impurity": None,
        "proportion": True,
        "feature_names": pipeline.steps[-2][1].columns_to_keep_,
    }
    if isinstance(tree_classifier, DecisionTreeClassifier):
        kwargs["class_names"] = tree_classifier.classes_

    dot_data = export_graphviz(tree_classifier, **kwargs)
    return graphviz.Source(dot_data)
