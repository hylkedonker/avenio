from copy import copy

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import unique_labels


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
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
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
