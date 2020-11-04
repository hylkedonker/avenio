from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from fragment_count.utils import filter_no_fragments, pool_timepoints, safe_normalise


def plot_distribution(seq, label=None, with_peaks=True):
    fragment_window = [80, 400]
    distribution = seq.copy() / sum(seq)
    distribution.plot(label=label, zorder=0)
    if with_peaks:
        n = 5
        peak_constaints = {"width": 15, "distance": 50, "rel_height": 0.85}
        peaks, properties = find_peaks(
            distribution.rolling(n).mean(), **peak_constaints
        )
        plt.plot(peaks, distribution[peaks], "o", zorder=1)
        plt.hlines(
            y=properties["width_heights"],
            xmin=properties["left_ips"],
            xmax=properties["right_ips"],
            color="C1",
            zorder=2,
        )
    plt.xlim(fragment_window)


def plot_distribution_errorbar(seq, label=None, with_peaks=True):
    """
    Plot distribution with patient variability.
    """
    # Combine baseline and follow-up.
    df = pool_timepoints(seq)
    # Remove patients with no fragments at both base line and follow up.
    df = filter_no_fragments(df)
    # Normalise as probability.
    distributions = df.groupby("Patient ID").apply(safe_normalise)

    mean = distributions.groupby("length (bp)").mean()
    std = distributions.groupby("length (bp)").std()
    fragment_window = [80, 400]
    print("integral", mean.sum())

    n = distributions.index.get_level_values("Patient ID").nunique()
    mean.plot(label=label + " (n={})".format(n))
    upper_bound = mean + std
    lower_bound = mean - std
    plt.fill_between(mean.index, upper_bound, lower_bound, alpha=0.25)

    plt.xlim(fragment_window)


def plot_motif(
    motifs,
    wild_type_frequency,
    variant_frequency,
    wild_type_error=None,
    variant_error=None,
    ax=None,
):
    """
    Make bar plot comparison between wild type frequencies and variant frequencies.
    """
    if ax is None:
        plt.rc("font", family="serif")
        plt.figure(figsize=(4, 3))
        ax = plt.gca()

    width = 0.25
    x = np.arange(len(motifs))
    wild_kwargs = {
        "color": "C0",
        "alpha": 0.75,
        "label": "wild-type",
    }
    if wild_type_error is not None:
        wild_kwargs["yerr"] = wild_type_error.loc[motifs]

    ax.bar(x - width / 2, wild_type_frequency.loc[motifs], width, **wild_kwargs)
    variant_kwargs = {
        "color": "C1",
        "alpha": 0.75,
        "label": "variant",
    }
    if variant_error is not None:
        variant_kwargs["yerr"] = variant_error.loc[motifs]
    ax.bar(x + width / 2, variant_frequency.loc[motifs], width, **variant_kwargs)
    ax.set_xticks(x)
    ax.set_xticklabels(motifs)
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Motif")

    plt.legend(frameon=False)


def plot_distribution_comparison(
    normal_counts, variant_counts, filename_suffix=None, labels=None
):
    normal_distribution = normal_counts.copy() / sum(normal_counts)
    normal_cumul = normal_distribution.cumsum()
    variant_distribution = variant_counts.copy() / sum(variant_counts)
    variant_cumul = variant_distribution.cumsum()

    fragment_window = [50, 500]

    label_normal = "normal (n={})".format(normal_counts.sum().sum())
    label_variant = "variant (n={})".format(variant_counts.sum().sum())
    if labels is not None:
        label_normal, label_variant = labels[:2]
    normal_distribution.plot(label=label_normal)
    variant_distribution.plot(label=label_variant)
    plt.legend(frameon=False)
    plt.xlim(fragment_window)

    plt.xlabel("Fragment size (bp)")
    plt.ylabel("Frequency")
    if filename_suffix is not None:
        plt.savefig(f"figs/fragment_size_distribution_{filename_suffix}.png")

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(normal_cumul, label=label_normal)
    plt.plot(variant_cumul, label=label_variant)
    plt.ylabel("Cumulative distribution")
    plt.xlabel("Fragment size (bp)")
    plt.xlim(fragment_window)

    plt.subplot(1, 2, 2)
    plt.plot(normal_cumul - variant_cumul)
    plt.xlabel("Fragment size (bp)")
    plt.ylabel("Kolmogorov-Smirnov distance")
    plt.xlabel("Fragment size (bp)")
    plt.xlim(fragment_window)
    plt.tight_layout()
    if filename_suffix is not None:
        plt.savefig(f"figs/fragment_size_distribution_difference_{filename_suffix}.png")
