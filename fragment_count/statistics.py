from numpy import exp

from fragment_count.utils import to_cumulative


def kolmogorov_smirnov(normal_counts, variant_counts):
    m = sum(normal_counts)
    n = sum(variant_counts)
    D = max(to_cumulative(normal_counts) - to_cumulative(variant_counts))
    r = (n + m) / (n * m)
    alpha = 2 * exp(-2 * D ** 2 / r)
    return alpha
