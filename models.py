from typing import Optional

from catboost import CatBoostClassifier
import gensim
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import safe_indexing, safe_mask

from const import target_genes
from utils import get_categorical_columns


class SparseFeatureFilter(BaseEstimator, TransformerMixin):
    """
    Filter out features that are non-zero less than a given thresshold.
    """

    def __init__(self, thresshold: int = 5):
        self.thresshold = thresshold

    def fit(self, X, y=None):
        """
        Find columns with at least `threshold` non-zero values.
        """
        # What elements are non-zero?
        non_zero_values = safe_mask(X, X != 0)

        # Pick columns that have at least `thresshold` occurences.
        above_thresshold = np.sum(non_zero_values, axis=0) >= self.thresshold

        # Get column names if pandas.
        if isinstance(X, pd.DataFrame):
            self.columns_to_keep_ = X.columns[above_thresshold].values
        # Otherwise the indices.
        else:
            self.columns_to_keep_ = np.nonzero(above_thresshold)[0]

        return self

    def transform(self, X, y=None):
        """
        Chuck out columns below thresshold.
        """
        if isinstance(X, pd.DataFrame):
            return X[self.columns_to_keep_]
        return X[:, self.columns_to_keep_]


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
            X, y=y, cat_features=self.cat_features, eval_set=self.eval_set, **fit_params
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


class Gene2Vec(BaseEstimator, TransformerMixin):
    """
    Transform gene columns to gene embeddings.

    X'(i,k) = sum_j X_(ij) * gene(j, k)
    with gene(j, k) the embedding vector.
    """

    def __init__(
        self,
        embedding_model: gensim.models.word2vec.Word2Vec = None,
        remainder: str = "ignore",
    ):
        # Load model if none was passed.
        if embedding_model is None:
            self.model = gensim.models.KeyedVectors.load_word2vec_format(
                "gene2vec_dim_200_iter_9_w2v.txt", binary=False
            )
        else:
            self.model = embedding_model

        if remainder not in ("ignore", "drop", "fail"):
            raise ValueError(f"Unknown action for remainder={remainder}.")
        self.unknown_columns_ = remainder

    def fit(self, X: pd.DataFrame, y=None):
        """
        Parse columns to process.
        """
        # We need the column names for the embeddings.
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be pandas data frame.")

        if (
            not set(X.columns).issubset(set(target_genes))
            and self.unknown_columns_ == "fail"
        ):
            raise KeyError(f"Unknown columns in {X.columns}")

        # What columns to use.
        self.columns_to_transform_ = tuple(
            column for column in X.columns if column in self.model
        )

        return self

    def transform(self, X, y=None):
        """
        Generate new matrix by summing embedding contributions.
        """
        # Allocate memory for storing the embeddings.
        X_T = np.zeros([X.shape[0], self.model.vector_size])

        for column in self.columns_to_transform_:
            # Embedding for given gene.
            v = self.model[column]
            # Multiply embedding by allele frequency (for each patient). Sum
            # contributions of all genes.
            X_T += X[column].values.reshape(-1, 1) * v.reshape(1, -1)

        # Keep remainder of columns when mode is "ignore".
        if self.unknown_columns_ == "ignore":
            columns_to_keep = [
                column
                for column in X.columns
                if column not in self.columns_to_transform_
            ]
            # When list not empty, append the transformed data.
            if columns_to_keep:
                return pd.concat(
                    [X[columns_to_keep], pd.DataFrame(X_T, index=X.index)], axis=1
                )

        # Otherwise just drop all other columns (= keep only transformed colums).
        return pd.DataFrame(X_T).copy()


class MergeRareCategories(BaseEstimator, TransformerMixin):
    """
    Merge categories occuring equal or less than `thresshold` times.
    """

    def __init__(
        self,
        categorical_columns: Optional[list] = None,
        thresshold: int = 30,
        unique_column: str = "raise",
    ):
        """
        Merge columns in `categorical_columns` occuring less than `thresshold`.

        Args:
            categorical_columns (list): Carry out transformation on all non-numeric
                columns when None are provided.
            unique_ (str): How to handle columns with more than 80 % unique
                values. Posibble values: {"raise", "ignore"}.
        """
        self.thresshold_ = thresshold
        self.unique_column_ = unique_column

        if not self.thresshold_:
            raise ValueError("No thresshold!")

        self.categorical_columns_ = categorical_columns

    def get_params(self, deep: bool = True):
        return {
            "thresshold": self.thresshold_,
            "categorical_columns": self.categorical_columns_,
        }

    def fit(self, X: pd.DataFrame, y=None):
        """
        Look for categories to merge.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be Pandas data frame.")

        # Keep track of categories, per column, that are to be merged.
        self.categories_to_merge_ = {}

        # Use non-numeric columns when None were provided.
        if not self.categorical_columns_:
            self.categorical_columns_ = get_categorical_columns(X)

        # Go through all the categorical columns.
        for column in self.categorical_columns_:
            if len(X[column].unique()) >= 0.8 * len(X[column]):
                # Raise error if there are too many unique categories (probably numeric
                # column).
                if self.unique_column_.lower() != "ignore":
                    raise KeyError(
                        (
                            r"More than 80 % of values in column `{}` are unique! "
                            "Probably not a categorical column."
                        ).format(column)
                    )

            for category in X[column].unique():
                # Check that each category occurs at least `thresshold` times.
                constraint = X[column] == category

                if len(X[constraint]) <= self.thresshold_:
                    # Add category to the merge list.
                    if column not in self.categories_to_merge_:
                        self.categories_to_merge_[column] = [category]
                    else:
                        self.categories_to_merge_[column].append(category)

            # At least two categories are needed to merge. Don't carry out a trivial
            # transformation for one column.
            if (
                column in self.categories_to_merge_
                and len(self.categories_to_merge_[column]) == 1
            ):
                del self.categories_to_merge_[column]

        return self

    def transform(self, X: pd.DataFrame, y=None):
        """
        Merge the categories.

        Merge policy: group all categories below the thresshold into one new (composite)
        category.
        """
        for column, category_list in self.categories_to_merge_.items():
            new_category_name = "+".join(str(cat) for cat in category_list)
            # Replace all of the cells belonging to any of the below-thresshold
            # categories with the new composite category.
            constraint = X[column].isin(category_list)
            X.loc[constraint, column] = new_category_name
            X[column] = X[column].astype(str)

        return X.copy()
