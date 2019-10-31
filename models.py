from typing import Optional

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder


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


class CustomCatBoostClassifier(CatBoostClassifier):
    def __init__(self, cat_features, eval_set=None, **kwargs):
        self.cat_features = cat_features
        self.eval_set = eval_set
        super().__init__(**kwargs)

    def fit(self, X, y=None, **fit_params):
        """
        Fit catboost classifier.
        """
        return super().fit(
            X,
            y=y,
            cat_features=self.cat_features,
            eval_set=self.eval_set,
            **fit_params
        )


class ClassifierAsTransformer(BaseEstimator, TransformerMixin):
    """
    Wrap transformer around classifier.
    """

    def __init__(self, classifier, encoder: Optional = OrdinalEncoder()):
        self.classifier = classifier
        self.encoder = encoder

    def _to_matrix(self, y):
        """
        Represent vector as matrix.
        """
        if hasattr(y, "shape"):
            if len(y.shape) == 1:
                if isinstance(y, (pd.Series, pd.DataFrame)):
                    y = y.to_numpy()
                y = y.reshape([-1, 1])
        else:
            y = np.array(y).reshape([-1, 1])

        return y

    def fit(self, X, y):
        self.classifier.fit(X, y)

        y = self._to_matrix(y)
        if self.encoder is not None:
            self.encoder.fit(y)

        return self

    def transform(self, X, y=None):
        """
        Redirect output from classifier.
        """
        y_output = self.classifier.predict(X)

        # Encode output of classifier.
        if self.encoder:
            y_output = self._to_matrix(y_output)
            y_output = self.encoder.transform(y_output)

        return y_output
