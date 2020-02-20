from copy import copy
from typing import Iterable, Optional

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
    calculate_pass_through_column_names_Freeman,
    reconstruct_categorical_variable_names,
)
from utils import bootstrap

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
    plt.style.use("default")
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    yticks = np.arange(cm.shape[0])
    xticks = np.arange(cm.shape[1])

    # We want to show all ticks...
    ax.set(
        xticks=xticks,
        yticks=yticks,
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
    )

    ax.set_ylabel("True label", fontsize="x-large")
    ax.set_xlabel("Predicted label", fontsize="x-large")
    ax.set_xlim([xticks[0] - 0.5, xticks[-1] + 0.5])
    ax.set_ylim([yticks[-1] + 0.5, yticks[0] - 0.5])
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", rotation_mode="anchor")
    # Rotate the tick labels and set their alignment.

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
                fontsize="xx-large",
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
        # Calculate signal.
        s = categorical_signal(df)
        # Add results to summary.
        s["item"] = s.index
        s["category"] = category
        # Add results to summary.
        summary = summary.append(s, ignore_index=True)

    return summary.set_index(["category", "item"])


def remove_parallel_coefficients(coeff_mean, coeff_std, names):
    """
    Remove coefficients that are (almost) equal but opposite in sign.
    """
    name_new, coef_mean_new, coef_std_new = [], [], []
    for i, y_i in enumerate(coeff_mean):
        if i < len(names) - 1:
            # If almost opposite in sign.
            if np.absolute(y_i + coeff_mean[i + 1]) < 0.01 * np.std(coeff_mean):
                continue

        name_new.append(names[i])
        coef_mean_new.append(y_i)
        coef_std_new.append(coeff_std[i])

    return coef_mean_new, coef_std_new, name_new


def remove_coefficients_below_thresshold(coeff_mean, coeff_std, names, thresshold=0.05):
    """
    Remove coefficients for which the magnitude |c_i| < thresshold.
    """
    name_new, coef_mean_new, coef_std_new = [], [], []
    for i, y_i in enumerate(coeff_mean):
        if abs(y_i) > thresshold:
            name_new.append(names[i])
            coef_mean_new.append(y_i)
            coef_std_new.append(coeff_std[i])

    return coef_mean_new, coef_std_new, name_new


@bootstrap(k=5)
def fit_model_coefficients(X_train, y_train, X_test, y_test, pipeline):
    """
    Fit coefficients only, ignore test data.

    Signature:
    fit_model_coefficients(X, y)
    """
    pipeline.fit(X_train, y_train)
    estimator = pipeline.named_steps["estimator"]
    return estimator.coef_.flatten()


def view_linear_model_richard(X, y, pipeline):
    """
    Plot the coefficients of Richard model.
    """
    # Calculate coefficients' mean and standard deviation of the bootstrapped model.
    bootstrapped_coefficients = fit_model_coefficients(X, y, pipeline)
    coeff_mean, coeff_std = bootstrapped_coefficients

    variable_names = []
    # Generate variable names of the one hot encoded categorical data.
    variable_names.extend(reconstruct_categorical_variable_names(pipeline))
    # Concatenate with unaltered phenotype columns.
    variable_names.extend(calculate_pass_through_column_names_Richard(pipeline))

    coeff_mean, coeff_std, variable_names = remove_parallel_coefficients(
        coeff_mean, coeff_std, variable_names
    )
    # print("Warning: Removed redundant coefficients", set(variable_names) - set(names))

    with sns.plotting_context(font_scale=1.5):
        plt.figure(figsize=(8, 6))
        plt.xlabel(r"Slope")
        sns.barplot(x=coeff_mean, xerr=coeff_std, y=variable_names, label="large")
        plt.tight_layout()


def view_linear_model_julian(X, y, p_julian):
    """
    Plot the coefficients of Richard model.
    """
    coeff_mean, coeff_std = fit_model_coefficients(X, y, p_julian)

    column_names = p_julian.steps[-2][1].columns_to__
    with sns.plotting_context(font_scale=1.5):
        plt.xlabel(r"Slope")
        sns.barplot(x=coeff_mean, xerr=coeff_std, y=column_names, label="large")
        plt.tight_layout()


def view_decision_tree_julian(pipeline, save_to: Optional[str] = None):
    """
    Plot the decision tree of a Julian pipeline.
    """
    tree_classifier = pipeline.steps[-1][1]
    # Consistency check of the Julian pipeline.
    assert isinstance(pipeline.steps[-2][1], SparseFeatureFilter)

    kwargs = {
        "max_depth": 3,
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
    g = graphviz.Source(dot_data)

    if save_to:
        filename, extension = save_to.split(".")
        g.render(filename, format=extension)

    return g


def view_linear_model_freeman(X, y, pipeline):
    """
    Infer the variable names and plot the coefficients.
    """
    # Calculate coefficients' mean and standard deviation of the bootstrapped model.
    bootstrapped_coefficients = fit_model_coefficients(X, y, pipeline)
    coeff_mean, coeff_std = bootstrapped_coefficients

    # Generate variable names of the one hot encoded categorical data.
    clinical_variable_names = reconstruct_categorical_variable_names(pipeline)
    # Concatenate with unaltered columns. The remaining variables are the genetic ones.
    genetic_variable_names = calculate_pass_through_column_names_Freeman(pipeline)

    # Seperate the variables in clinical variables, and genetic variables.
    number_clinical_vars = len(clinical_variable_names)
    coeff_mean_clinical = coeff_mean[:number_clinical_vars]
    coeff_std_clinical = coeff_std[:number_clinical_vars]
    coeff_mean_genetic = coeff_mean[number_clinical_vars:]
    coeff_std_genetic = coeff_std[number_clinical_vars:]

    # Make a plot for the clinical data.
    with sns.plotting_context(font_scale=1.5):
        # Remove uninformative conjugate variables, due to one-hot-encoding.
        coeff_mean_clinical, coeff_std_clinical, clinical_variable_names = remove_parallel_coefficients(
            coeff_mean_clinical, coeff_std_clinical, clinical_variable_names
        )
        coeff_mean_clinical, coeff_std_clinical, clinical_variable_names = remove_coefficients_below_thresshold(
            coeff_mean_clinical,
            coeff_std_clinical,
            clinical_variable_names,
            thresshold=0.05,
        )

        plt.figure(figsize=(8, 6))
        plt.title("Clinical variables")
        plt.xlabel(r"Slope $c_i$ ($\|c_i\| > 0.05$)")
        sns.barplot(
            x=coeff_mean_clinical,
            xerr=coeff_std_clinical,
            y=clinical_variable_names,
            label="large",
        )
        plt.tight_layout()
        plt.savefig(
            "figs/logistic_regression_clinical_freeman.png", bbox_inches="tight"
        )
        plt.savefig(
            "figs/logistic_regression_clinical_freeman.eps", bbox_inches="tight"
        )

    # And a seperate figure for the genetic data.
    with sns.plotting_context(font_scale=1.5):
        # Remove very small coefficients.
        coeff_mean_genetic, coeff_std_genetic, genetic_variable_names = remove_coefficients_below_thresshold(
            coeff_mean_genetic,
            coeff_std_genetic,
            genetic_variable_names,
            thresshold=0.05,
        )
        plt.figure(figsize=(8, 6))
        plt.title("Genetic variables")
        plt.xlabel(r"Slope $c_i$ ($\|c_i\| > 0.05$)")
        sns.barplot(
            x=coeff_mean_genetic,
            xerr=coeff_std_genetic,
            y=genetic_variable_names,
            label="large",
        )
        plt.tight_layout()
        plt.savefig("figs/logistic_regression_genetic_freeman.png", bbox_inches="tight")
        plt.savefig("figs/logistic_regression_genetic_freeman.eps", bbox_inches="tight")


# from sklearn.linear_model import LogisticRegression
# from transform import combine_tsv_files
# from pipelines import benchmark_pipelines, build_classifier_pipelines, pipeline_Freeman

# # Harmonic mean genomic variable.
# X_train_hm, y_train_hm = combine_tsv_files(
#     "output/train__harmonic_mean__Allele Fraction.tsv",
#     "output/train__harmonic_mean__CNV Score.tsv",
# )
# y_train_resp = y_train_hm["response_grouped"]
# logistic_Freeman = pipeline_Freeman(LogisticRegression)
# view_linear_model_freeman(X_train_hm, y_train_resp, logistic_Freeman)
