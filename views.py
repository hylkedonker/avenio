from copy import copy
from typing import Dict, Iterable, Optional, Tuple

import graphviz
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from transform import combine_tsv_files
from models import SparseFeatureFilter
from pipelines import (
    calculate_pass_through_column_names_Richard,
    calculate_pass_through_column_names_Freeman,
    get_hyper_param_grid,
    nested_cross_validate_score,
    reconstruct_categorical_variable_names,
)
from utils import bootstrap

# matplotlib.rc("font", size=22)
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


def dichomotise_parallel_coefficients(coeff_mean, coeff_std, names):
    """
    Partition coefficients in two groups that are equal but opposite in sign.
    """
    name_a, coef_mean_a, coef_std_a = [], [], []
    name_b, coef_mean_b, coef_std_b = [], [], []

    for i, y_i in enumerate(coeff_mean):
        if i < len(names) - 1:
            # If almost opposite in sign.
            if np.absolute(y_i + coeff_mean[i + 1]) < 0.01 * np.std(coeff_mean):
                # Add to b instead.
                name_b.append(names[i])
                coef_mean_b.append(y_i)
                coef_std_b.append(coeff_std[i])
                continue

        name_a.append(names[i])
        coef_mean_a.append(y_i)
        coef_std_a.append(coeff_std[i])

    return (coef_mean_a, coef_std_a, name_a), (coef_mean_b, coef_std_b, name_b)


def remove_coefficients_below_thresshold(data_frame, thresshold=0.05):
    """
    Remove coefficients for which the magnitude |c_i| < thresshold.
    """
    return data_frame.loc[abs(data_frame["mean"]) > thresshold]


@bootstrap(k=5)
def fit_estimator_coefficients(X_train, y_train, X_test, y_test, pipeline):
    """
    Fit estimator coefficients only (locked preprocessing pipeline), ignore test data.

    Use cross validation to determine the optimal hyper parameters.

    Signature:
    fit_model_coefficients(X, y)
    """
    # Transform the data using the locked preprocessing pipeline.
    preprocess_pipeline = Pipeline(pipeline.steps[:-1])
    X_train_transf = preprocess_pipeline.transform(X_train)

    # Fit the coefficients of the estimator only using cross validaiton. But don't refit
    # the remaining pipeline.
    estimator = pipeline.named_steps["estimator"]

    hyper_parameters = get_hyper_param_grid(estimator)
    inner_loop = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
    clf = GridSearchCV(estimator, param_grid=hyper_parameters, cv=inner_loop, n_jobs=-1)
    clf.fit(X_train_transf, y_train)

    return clf.best_estimator_.coef_.flatten()


def fit_model_coefficients(X, y, pipeline):
    """
    Fit the coefficients of the model (ignore the test data).
    """
    # First fit the preprocessing pipeline to all the data. This ensures that all the
    # one-hot-encoding categories have been picked up.
    pipeline.fit(X, y)
    # Lock the pipeline up to the estimator. That is, fit only the estimator (using
    # bootstrapping) -- don't refit the remaining pipeline.
    return fit_estimator_coefficients(X, y, pipeline)


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


def merge_partitioned_coefficients(partition_a: tuple, partition_b: tuple):
    """
    Merge complementary variables into single representation.
    """

    def _unpack_feature_category_names(coefficient_names):
        """  """
        feature_category_pairs = (
            coef_name.split(":") for coef_name in coefficient_names
        )
        return zip(*feature_category_pairs)

    # Unpack feature-category pairs.
    feature_names_a, category_names_a = _unpack_feature_category_names(partition_a[2])
    feature_names_b, category_names_b = _unpack_feature_category_names(partition_b[2])

    # The features must be paired.
    if feature_names_a != feature_names_b:
        raise ValueError(
            "Left hand feature partition is not aligned with right hand partition."
        )

    # Create new variable name.
    new_names = map(
        lambda x: f"{x[0]}: {x[1].strip()} vs. {x[2].strip()}",
        zip(feature_names_a, category_names_a, category_names_b),
    )
    new_mean = map(lambda x: x[0] - x[1], zip(partition_a[0], partition_b[0]))
    new_std = np.array(partition_a[1]) * 2

    new_mean = np.fromiter(new_mean, dtype=float)

    return (new_mean, new_std, np.array(tuple(new_names)))


def view_linear_model_freeman(X, y, pipeline, top_n=10, filenames=None):
    """
    Infer the variable names and plot the coefficients.
    """
    assert len(np.unique(y)) == 2

    # Calculate coefficients' mean and standard deviation of the bootstrapped model.
    bootstrapped_coefficients = fit_model_coefficients(X, y, pipeline)
    coeff_mean, coeff_std = bootstrapped_coefficients

    # Generate variable names from the one-hot-encoded categorical data.
    clinical_variable_names = reconstruct_categorical_variable_names(pipeline)

    # Concatenate with unaltered columns. The remaining variables are the genetic ones.
    passthrough_columns = calculate_pass_through_column_names_Freeman(pipeline)

    # Seperate the variables in clinical variables, and genetic variables.
    number_clinical_vars = len(clinical_variable_names)
    coeff_mean_clinical = coeff_mean[:number_clinical_vars]
    coeff_std_clinical = coeff_std[:number_clinical_vars]

    coeff_mean_genetic = coeff_mean[number_clinical_vars:]
    coeff_std_genetic = coeff_std[number_clinical_vars:]

    genetic_variable_names = passthrough_columns

    assert len(clinical_variable_names) + len(genetic_variable_names) == len(coeff_mean)

    # Make a plot for the clinical data.
    with sns.plotting_context(font_scale=1.5):
        coef_partitions = dichomotise_parallel_coefficients(
            coeff_mean_clinical, coeff_std_clinical, clinical_variable_names
        )

        def _init_plot():
            """ Initialise clinical plot settings. """
            plt.figure(figsize=(8, 6))
            plt.title("Clinical variables")
            plt.xlabel(r"Slope difference $\Delta c_i$")

        try:
            merged_mean, merged_std, merged_name = merge_partitioned_coefficients(
                coef_partitions[1], coef_partitions[0]
            )
        except ValueError:
            # When unable to merge, generate two indepdendent plots.
            for (
                i,
                (coeff_mean_clinical, coeff_std_clinical, clinical_variable_names),
            ) in enumerate(coef_partitions):
                _init_plot()
                sns.barplot(
                    x=coeff_mean_clinical,
                    xerr=coeff_std_clinical,
                    y=clinical_variable_names,
                    label="large",
                )
                plt.tight_layout()
                if filenames:
                    plt.savefig(
                        "figs/{}_{}.png".format(filenames[0], i + 1),
                        bbox_inches="tight",
                    )
                    plt.savefig(
                        "figs/{}_{}.eps".format(filenames[0], i + 1),
                        bbox_inches="tight",
                    )
        else:
            _init_plot()
            sns.barplot(x=merged_mean, xerr=merged_std, y=merged_name, label="large")
            plt.tight_layout()
            if filenames:
                plt.savefig("figs/{}.png".format(filenames[0]), bbox_inches="tight")
                plt.savefig("figs/{}.eps".format(filenames[0]), bbox_inches="tight")

        plot1_limits = plt.xlim()

    # And a seperate figure for the genetic data.
    with sns.plotting_context(font_scale=1.5):
        coef_data_frame = pd.DataFrame(
            {"mean": coeff_mean_genetic, "std": coeff_std_genetic},
            index=genetic_variable_names,
        )

        coef_data_frame["mean_magnitude"] = abs(coef_data_frame["mean"])
        max_n = top_n
        if top_n is None:
            max_n = coeff_mean_genetic.shape[0] + 1

        top_n_coef = coef_data_frame.sort_values(
            by="mean_magnitude", ascending=False
        ).iloc[:max_n]

        # plt.figure(figsize=(6, 8))
        plt.figure()
        if coef_data_frame.shape[0] > max_n:
            plt.title(f"Top-{max_n} genetic variables")
        from matplotlib import rc

        rc("text.latex", preamble=r"\usepackage{xcolor}")

        def change_variant_label(label_input):
            gene, variation = label_input.split("_")
            if variation == "cnv":
                gene_text = r"\bf{" + gene + "}"
            else:
                gene_text = gene
            return r"$\mathrm{" + gene_text + r"_{" + variation + r"}}$"

        top_n_coef["labels"] = top_n_coef.index.to_series().apply(change_variant_label)
        c0, c1 = sns.color_palette()[:2]
        sns.barplot(
            x=top_n_coef["mean"],
            xerr=top_n_coef["std"],
            y=top_n_coef["labels"],
            palette=top_n_coef["labels"].apply(lambda x: c0 if "cnv" in x else c1),
            label="large",
            color="gray",
        )
        # plt.ylabel("Gene")
        plt.xlabel(r"Slope $c_i$ (-)")
        plt.xlim(plot1_limits)
        plt.tight_layout()
        if filenames:
            plt.savefig("figs/{}.png".format(filenames[1]), bbox_inches="tight")
            plt.savefig("figs/{}.eps".format(filenames[1]), bbox_inches="tight")


def compare_prognostic_value_genomic_information(
    feature_label_pairs: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]], plot_label=None
):
    """
    feature_label_pairs: Pairs of (model, (X, y)) to make a comparison.
    """
    results = pd.DataFrame(index=feature_label_pairs.keys(), columns=["mean", "std"])
    for label, (model, (X, y)) in feature_label_pairs.items():
        scores = nested_cross_validate_score(model, X, y, metric="roc_auc")
        results.loc[label, "mean"] = np.mean(scores)
        results.loc[label, "std"] = np.std(scores)
    plt.errorbar(
        x=results.index, y=results["mean"], yerr=results["std"], label=plot_label
    )
    degrees = 90
    plt.xticks(rotation=degrees)
    plt.ylim([0.5, 1.0])
    plt.ylabel("AUC ROC")
    plt.legend(frameon=False)


# from sklearn.linear_model import LogisticRegression
# from transform import combine_tsv_files
# from pipelines import benchmark_pipelines, build_classifier_pipelines, pipeline_Freeman

# # Harmonic mean genomic variable.
# X_train_hm, y_train_hm = combine_tsv_files(
#     "output/train__harmonic_mean__Allele Fraction.tsv",
#     "output/train__harmonic_mean__CNV Score.tsv",
# )
# pos_label = 'responder (pr+cr)'
# y_train_resp = y_train_hm["response_grouped"]  == pos_label
# parameters = {"solver": "newton-cg"}
# logistic_Freeman = pipeline_Freeman(LogisticRegression, **parameters)
# view_linear_model_freeman(X_train_hm, y_train_resp, logistic_Freeman, thresshold=0.05)

# from sklearn.linear_model import LogisticRegression
# from transform import combine_tsv_files
# from pipelines import (
#     benchmark_pipelines,
#     build_classifier_pipelines,
#     pipeline_Freeman,
#     pipeline_Richard,
# )


# parameters = {
#     "C": 0.1,
#     "class_weight": "balanced",
#     "solver": "newton-cg",
# }
# mutant_data_pairs = generate_data_pairs(
#     filename_prefix="output/train", snv_type="No. Mutant Molecules per mL"
# )
# vaf_data_pairs = generate_data_pairs(
#     filename_prefix="output/train", snv_type="Allele Fraction"
# )
# model_mutant_data_pairs = generate_model_data_pairs(mutant_data_pairs, parameters)
# model_vaf_data_pairs = generate_model_data_pairs(vaf_data_pairs, parameters)
# compare_prognostic_value_genomic_information(model_mutant_data_pairs, plot_label="Mutant concentration")
# compare_prognostic_value_genomic_information(model_vaf_data_pairs, plot_label='Allele fraction')
# plt.savefig('figs/comparison_genomic_data.png', bbox_inches="tight")
