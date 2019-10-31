from sklearn.base import BaseEstimator, TransformerMixin


class UniqueFeatureFilter(BaseEstimator, TransformerMixin):
    """
    Filter out features that are non-zero more often than a given thresshold.
    """

    def __init__(self, thresshold: int = 5):
        self.thresshold = thresshold

    def fit(self, X, y=None):
        """
        Find columns with at least `threshold` non-zero values.
        """
        # What elements are non-zero?
        non_zero_values = X[X != 0]
        # Pick columns that have at least `thresshold` occurences.
        above_thresshold = non_zero_values.count() >= self.thresshold
        self.columns_to_keep = above_thresshold.index[above_thresshold].values

        return self

    def transform(self, X, y=None):
        """
        Chuck out columns below thresshold.
        """
        return X[self.columns_to_keep]
