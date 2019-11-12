import unittest

import gensim
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from models import (
    ClassifierAsTransformer,
    Gene2Vec,
    MergeRareCategories,
    UniqueFeatureFilter,
)


class TestUniqueFeatureFilter(unittest.TestCase):
    def test_filter(self):
        """
        Test thessholding feature filter.
        """
        X = pd.DataFrame(
            np.array([range(4), [0] * 3 + [1], [0] * 2 + [1, 2]]).T,
            columns=["a", "b", "c"],
        )
        f = UniqueFeatureFilter(thresshold=2)
        f.fit(X)
        # Check that correct columns are filtered.
        self.assertEqual(set(f.columns_to_keep), {"a", "c"})
        # Test that array is correctly transformed.
        np.testing.assert_array_equal(f.transform(X), X[["a", "c"]])


class TestClassifierAsTransformer(unittest.TestCase):
    def setUp(self):
        """
        Initialise environment for testing.
        """
        self.seed = 1234
        np.random.seed(self.seed)

    def test_pipelines(self):
        """
        Verify that the transformer wrapper classifier works as expected.
        """
        X = np.random.random([10, 2])
        y = ["PD", "SD", "PR", "PD", "PR", "SD", "CR", "PD", "PR", "SD"]
        tree = DecisionTreeClassifier(random_state=self.seed).fit(X, y)
        tree_transformer = ClassifierAsTransformer(
            classifier=DecisionTreeClassifier(random_state=self.seed), encoder=None
        ).fit(X, y)
        np.testing.assert_array_equal(tree.predict(X), tree_transformer.transform(X))


class TestGene2Vec(unittest.TestCase):
    def setUp(self):
        # Load embedding.
        self.model = gensim.models.KeyedVectors.load_word2vec_format(
            "gene2vec_dim_200_iter_9_w2v.txt", binary=False
        )

    def test_embedding(self):
        """
        Test that genes are correctly embedded.
        """
        #
        X = pd.DataFrame(
            {"TP53": [1.0, 0.0, 2.0], "KRAS": [0.0, 2.0, 1.0], "age": [30, 40, 40]},
            index=[1, 2, 3],
        )

        with self.assertRaises(KeyError):
            Gene2Vec(embedding_model=self.model, remainder="fail").fit_transform(X)

        X_embed = Gene2Vec(
            embedding_model=self.model, remainder="ignore"
        ).fit_transform(X)

        # Check that the age column is unaffected.
        pd.testing.assert_series_equal(X_embed["age"], X["age"])

        # Check that the columns are correctly embedded.
        np.testing.assert_array_equal(X_embed.iloc[0, 1:], self.model["TP53"])
        # Second patient is a multiple of the vector.
        np.testing.assert_array_equal(X_embed.iloc[1, 1:], 2 * self.model["KRAS"])
        # Third patient is linear combination of the two.
        np.testing.assert_array_almost_equal(
            X_embed.iloc[2, 1:], 2 * self.model["TP53"] + self.model["KRAS"]
        )

        # Check that unknown columns are dropped, when specified by `unknown_columns`.
        X_embed2 = Gene2Vec(embedding_model=self.model, remainder="drop").fit_transform(
            X
        )
        self.assertEqual(X_embed2.shape[1], 200)


class TestMergeRareCategories(unittest.TestCase):
    def setUp(self):
        self.data_frame = pd.DataFrame(
            {
                "a": ["a1", "a2", "a1", "a2", "a1", "a3", "a4", "a4"],
                "b": ["a", "b"] * 4,
                # By default, not picked up as being categorical data.
                "c": ["1", "2", "1", "2", "1", "3", "4", "4"],
            }
        )

    def test_fit(self):
        """
        Check that correct categories are singled out.
        """
        # Check that an error is raised when non-categorical data is fed.
        numerical_data_frame = pd.DataFrame({"a": range(10)}).T
        with self.assertRaises(KeyError):
            MergeRareCategories(categorical_columns=["a"]).fit(numerical_data_frame)

        merger = MergeRareCategories(thresshold=2).fit(self.data_frame)
        self.assertEqual(tuple(merger.categories_to_merge_.keys()), ("a",))
        self.assertEqual(set(merger.categories_to_merge_["a"]), {"a2", "a3", "a4"})

        # Merge only column "c".
        specific_cat_merger = MergeRareCategories(
            thresshold=2, categorical_columns=["c"]
        ).fit(self.data_frame)
        self.assertEqual(tuple(specific_cat_merger.categories_to_merge_.keys()), ("c",))
        self.assertEqual(
            set(specific_cat_merger.categories_to_merge_["c"]), {"2", "3", "4"}
        )

    def test_transform(self):
        """
        Check that cells are correctly substituted.
        """
        # Test merge only "c".
        merger = MergeRareCategories(thresshold=2, categorical_columns=["c"])
        data_frame_transformed = merger.fit_transform(self.data_frame.copy())
        # Test that column "c" is correctly transformed.
        np.testing.assert_array_equal(
            np.array(["1", "2+3+4", "1", "2+3+4", "1", "2+3+4", "2+3+4", "2+3+4"]),
            data_frame_transformed["c"],
        )
        # Check that the remaining columns have been left untouched.
        np.testing.assert_array_equal(
            data_frame_transformed.iloc[:, :2], self.data_frame.iloc[:, :2]
        )

        # Let the transformer guess all categorical data (i.e., columns with non-numeric
        # data).
        default_cats_transformed = MergeRareCategories(thresshold=2).fit_transform(
            self.data_frame
        )

        np.testing.assert_array_equal(
            np.array(
                [
                    ["a1", "a", "1"],
                    ["a2+a3+a4", "b", "2"],
                    ["a1", "a", "1"],
                    ["a2+a3+a4", "b", "2"],
                    ["a1", "a", "1"],
                    ["a2+a3+a4", "b", "3"],
                    ["a2+a3+a4", "a", "4"],
                    ["a2+a3+a4", "b", "4"],
                ]
            ),
            default_cats_transformed,
        )
