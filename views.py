import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


# matplotlib.rc("font", size=22)
# matplotlib.rc("lines", linewidth=4)
# matplotlib.rc("figure", autolayout=True)
# matplotlib.rc("ytick", labelsize="large")
# matplotlib.rc("xtick", labelsize="large")
# matplotlib.rc("axes", labelsize="xx-large", titlesize="xx-large")
# matplotlib.rc("legend", fontsize="x-large")


def _confusion_matrix_plot(metrics_mean, metrics_std, ax, labels):
    """
    Plot the average confusion matrix with error bars.
    """
    confusion_matrix_mean = pd.DataFrame(
        metrics_mean["confusion_matrix"], index=labels, columns=labels
    )
    confusion_matrix_std = pd.DataFrame(
        metrics_std["confusion_matrix"], index=labels, columns=labels
    )
    c_annot = confusion_matrix_mean.applymap(
        lambda x: "{:0.2f}$\pm$".format(x)
    ) + confusion_matrix_std.applymap(lambda x: "{:0.2f}".format(x))
    sns.heatmap(
        confusion_matrix_mean,
        annot=c_annot,
        fmt="",
        cmap=plt.cm.Blues,
        ax=ax,
        cbar=False,
    )
    #     ax.set_yticks(rotation=0)
    ax.set_title(
        r"Accuracy {:0.2f}$\pm${:0.2f}".format(
            metrics_mean["accuracy"], metrics_std["accuracy"],
        )
    )
    ax.set_ylabel("Actual", weight="bold")
    ax.set_xlabel("Predicted", weight="bold")
    return ax


def _unpack_curve(mean, std, metric):
    if metric == "roc":
        x, y = mean["fprs"], mean["tprs"]
        dy = std["tprs"]
    elif metric == "precision_recall":
        x, y = mean["recall"], mean["precision"]
        dy = std["precision"]
    y_upper = np.minimum(y + dy, 1)
    y_lower = np.maximum(y - dy, 0)
    return x, y, y_lower, y_upper


def plot_roc_curve(m_mean, m_std, labels=None):
    fpr, tpr, tprs_lower, tprs_upper = _unpack_curve(m_mean, m_std, metric="roc")
    ax = plt.gca()
    plt.rc("font", family="serif")
    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title=f"Receiver operating characteristic",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", alpha=0.8)
    mean_label = r"Mean ROC (AUC = {:0.2f}$\pm${:0.2f})".format(
        m_mean["roc_auc"], m_std["roc_auc"]
    )
    print("Accuracy:", m_mean[f"accuracy"])
    ax.plot(fpr, tpr, label=mean_label, lw=2)
    ax.fill_between(
        fpr, tprs_lower, tprs_upper, color="grey", alpha=0.2, label=r"$\pm$ 1 std. dev."
    )
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(frameon=False, loc=2)
    inset_ax = ax.inset_axes([0.63, 0.15, 0.35, 0.35])
    if labels is None:
        labels = ["0", "1"]
    _confusion_matrix_plot(m_mean, m_std, inset_ax, labels=labels)


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


def plot_precision_recall(metrics: dict, labels=None):
    """
    Concatenate cross validation results.

    Ref: Forman, George, and Martin Scholz.
        "Apples-to-apples in cross-validation studies: pitfalls in classifier performance measurement.",
        Acm Sigkdd Explorations Newsletter 12.1 (2010): 49-57.
    """
    plt.figure(figsize=(4, 3))
    plt.rc("font", family="serif")
    # fig, ax = plt.subplots()
    ax = plt.gca()
    ax.set(
        xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=f"Precision-recall curve",
    )
    label = r"AUC = {:0.2f}".format(metrics["average_precision"])
    ax.plot(metrics["recall"], metrics["precision"], label=label, lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(frameon=False, loc="best")
    ax.set_xlim([0, 1])

