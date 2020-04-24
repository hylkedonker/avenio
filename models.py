from typing import Callable, Optional, Union

from catboost import CatBoostClassifier
import gensim
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MaxAbsScaler, OrdinalEncoder
from sklearn.utils import safe_mask

from const import target_genes
from utils import get_categorical_columns


class AggregateColumns(BaseEstimator, TransformerMixin):
    """
    Apply aggregation to a set of columns.
    """

    def __init__(
        self,
        columns: list,
        aggregate_function: Callable,
        aggregate_column_name: Optional[str] = None,
    ):
        self.columns_ = columns
        self.aggregate_function_ = aggregate_function

        self.aggregate_column_name_ = aggregate_column_name
        if self.aggregate_column_name_ is None:
            self.aggregate_column_name_ = aggregate_function.__name__

    def fit(self, X, y=None):
        """
        Check consistency of variables.
        """
        if not isinstance(X, pd.DataFrame):
            raise NotImplementedError("Does not yet support non-pandas inputs.")

        if set(self.columns_).isdisjoint(set(X.columns)):
            raise KeyError("No overlap in columns with `X`.")

        return self

    def transform(self, X, y=None):
        """
        Apply the transformation column-wise.
        """
        # Columns that should be left untouched in the transformation.
        passthrough_columns = list(set(X.columns) - set(self.columns_))
        passthrough_columns.sort()

        X_transformed = X[passthrough_columns].copy()
        X_transformed[self.aggregate_column_name_] = X[self.columns_].apply(
            self.aggregate_function_, axis=1
        )

        # Keep track of returned columns of the last transformation.
        self.returned_columns_ = passthrough_columns + [self.aggregate_column_name_]
        return X_transformed


class SparseFeatureFilter(BaseEstimator, TransformerMixin):
    """
    Filter out features that are non-zero less than a given thresshold.
    """

    def __init__(
        self,
        top_k_features: Optional[int] = None,
        thresshold: Optional[int] = None,
        columns_to_consider: Union[str, list] = "all",
    ):
        if top_k_features and thresshold:
            raise ValueError(
                "Choose either `top_k_features` or `thresshold`, not both."
            )
        elif not top_k_features and not thresshold:
            raise ValueError("Either set `top_k_features` or `thresshold`.")

        self.thresshold = thresshold
        self.top_k_features = top_k_features
        self.columns_to_consider = columns_to_consider

    def fit(self, X, y=None):
        """
        Filter out columns that do not meet the sparsity constraint.
        """
        # What elements are non-zero?
        non_zero_values = safe_mask(X, X != 0)

        # Number of columns not zero.
        non_zero_count = np.sum(non_zero_values, axis=0)

        # Pick columns that have at least `thresshold` occurences.
        if self.thresshold:
            below_thresshold = (
                non_zero_count < self.thresshold
            )  # Get column names if pandas.

            if isinstance(X, pd.DataFrame):
                self.columns_to_filter_ = X.columns[below_thresshold].values
            # Otherwise the indices.
            else:
                self.columns_to_filter_ = np.nonzero(below_thresshold)[0]
            # Filter out columns which should not be considered.
            if self.columns_to_consider != "all":
                self.columns_to_filter_ = list(
                    filter(
                        lambda x: x in self.columns_to_consider, self.columns_to_filter_
                    )
                )
        # Otherwise take the `k` largest columns (implicit thressholding).
        else:
            self.columns_to_filter_ = np.argsort(non_zero_count)
            # Filter out columns which should not be considered.
            # N.B.: This should be done before taking the top `k` columns. Otherwise we
            # end up with less than `k` features.
            if self.columns_to_consider != "all":
                if isinstance(X, pd.DataFrame):

                    def column_subset_filter(x):
                        return X.columns[x] in self.columns_to_consider

                else:

                    def column_subset_filter(x):
                        return x in self.columns_to_consider

                self.columns_to_filter_ = list(
                    filter(column_subset_filter, self.columns_to_filter_)
                )
            # After filtering out columns that should not be considered, take the top
            # `k` columns.
            self.columns_to_filter_ = self.columns_to_filter_[: -self.top_k_features]

            # Re-order columns in ascending order.
            self.columns_to_filter_ = sorted(self.columns_to_filter_)

            # Turn into column names, when Data Frame is passed.
            if isinstance(X, pd.DataFrame):
                self.columns_to_filter_ = X.columns[self.columns_to_filter_].values

        if isinstance(X, pd.DataFrame):
            self.columns_to_keep_ = list(
                filter(lambda x: x not in self.columns_to_filter_, X.columns)
            )
        else:
            self.columns_to_keep_ = list(
                filter(
                    lambda x: x not in self.columns_to_filter_, np.arange(0, X.shape[1])
                )
            )

        return self

    def transform(self, X, y=None):
        """
        Chuck out columns below thresshold.
        """
        if isinstance(X, pd.DataFrame):
            return X[self.columns_to_keep_]
        else:
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
        verify_categorical_columns: bool = True,
    ):
        """
        Merge columns in `categorical_columns` occuring less than `thresshold`.

        Args:
            categorical_columns (list): Carry out transformation on all non-numeric
                columns when None are provided.
            unique_ (str): How to handle columns with more than 80 % unique
                values. Posibble values: {"raise", "ignore"}.
            verify_categorical_columns (bool): Check that all categorical columns are
                present in the training data.
        """
        self.thresshold_ = thresshold
        self.unique_column_ = unique_column
        self.verify_categorical_columns_ = verify_categorical_columns

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

        # Auto determine the categorical columns (=non-numeric columns) when None.
        if self.categorical_columns_ is None:
            self.categorical_columns_ = get_categorical_columns(X)
        # Check that all columns are actually in the data frame.
        elif not set(self.categorical_columns_).issubset(set(X.columns)):
            if self.verify_categorical_columns_:
                raise KeyError(
                    "Some columns in in {} are not in X.".format(
                        self.categorical_columns_
                    )
                )
            # Check if there is at least some overlap.
            elif set(self.categorical_columns_).isdisjoint(set(X.columns)):
                raise KeyError(
                    "None of the supplied categorical columns are present in `X`."
                )

            # Perform transformation only for the overlap in columns.
            self.categorical_columns_ = list(
                set(self.categorical_columns_).intersection(set(X.columns))
            )
            self.categorical_columns_.sort()

        # Keep track of categories, per column, that are to be merged.
        self.categories_to_merge_ = {}

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
        X = X.copy()
        for column, category_list in self.categories_to_merge_.items():
            new_category_name = "+".join(str(cat) for cat in category_list)
            # Replace all of the cells belonging to any of the below-thresshold
            # categories with the new composite category.
            constraint = X[column].isin(category_list)
            X.loc[constraint, column] = new_category_name
            X.loc[:, column] = X[column].astype(str)

        return X


class AutoMaxScaler(BaseEstimator, TransformerMixin):
    """
    Determine non-categorical columns and max scale the values.
    """

    def __init__(
        self, ignore_columns: list = [], uniqueness_thresshold: Optional[float] = 0.8
    ):
        """
        Args:
            uniqueness_thresshold: Columns with less unique values than this
                are considered categorical.
        """
        self.ignore_columns = ignore_columns
        self.uniqueness_thresshold = uniqueness_thresshold

    def fit(self, X, y=None):
        """
        Determine which columns to min-max scale.
        """
        self.scaler_ = MaxAbsScaler(copy=True)
        categorical_columns = get_categorical_columns(X, self.uniqueness_thresshold)

        def is_numeric_and_not_ignored(column):
            if column not in categorical_columns and column not in self.ignore_columns:
                return True
            return False

        self.columns_to_transform_ = list(filter(is_numeric_and_not_ignored, X.columns))
        self.scaler_.fit(X[self.columns_to_transform_])
        return self

    def transform(self, X, y=None):
        """
        Max scale the columns and return copy.
        """
        data_subframe = X[self.columns_to_transform_]
        X[self.columns_to_transform_] = self.scaler_.transform(data_subframe)
        return X.copy()
