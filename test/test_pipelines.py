import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from pipelines import (
    pipeline_Freeman,
    reconstruct_categorical_variable_names,
    select_phenotype_columns,
    select_no_phenotype_columns,
)
from transform import combine_tsv_files


class TestPipelineUtils(unittest.TestCase):
    def test_filter_functions(self):
        """
        Test the phenotype filtering functions.
        """
        X = pd.DataFrame(
            # Generate 4x4 matrix.
            np.array(range(4 ** 2)).reshape(4, 4),
            columns=["a", "therapyline", "c", "lungmeta"],
        )
        # Test that `select_phenotype_columns` filters out all other columns.
        np.testing.assert_array_equal(
            select_phenotype_columns(X), X[["therapyline", "lungmeta"]]
        )
        # And test that `select_no_phenotype_columns` does precisely the
        # opposite.
        np.testing.assert_array_equal(select_no_phenotype_columns(X), X[["a", "c"]])

    def test_reconstruct_categorical_variable_names(self):
        """
        Test the variable name coding for the Freeman pipeline on real world data.
        """
        # Harmonic mean genomic variable.
        X_train_hm, y_train_hm = combine_tsv_files(
            "output/train__harmonic_mean__Allele Fraction.tsv",
            "output/train__harmonic_mean__CNV Score.tsv",
        )
        logistic_params = {
            "random_state": 1234,
            "penalty": "l2",
            "class_weight": "balanced",
            "solver": "newton-cg",
            "C": 1.0,
            "max_iter": 1000,
            "tol": 0.00001,
        }

        logistic_Freeman = pipeline_Freeman(LogisticRegression, **logistic_params)
        logistic_Freeman.fit(X_train_hm, y_train_hm["response_grouped"])

        # Calculate the names of the variables.
        variable_names = reconstruct_categorical_variable_names(logistic_Freeman)

        # Remove the estimator, so that we get the transformed data.
        del logistic_Freeman.steps[-1]
        X_transformed = logistic_Freeman.transform(X_train_hm)

        # Now check that the mapping is correct for a few variables.
        #
        # 1. Gender.
        males = X_transformed[X_train_hm["gender"] == "male"]
        reconstructed_index = variable_names.index("gender: male")
        np.testing.assert_array_equal(
            males[:, reconstructed_index], np.ones(males.shape[0])
        )

        # 2. Liver metastases.
        #
        # Check both for the catergory of presence, and the absence.
        no_liver_metas = X_transformed[
            X_train_hm["livermeta"] == "no metastasis present"
        ]
        liver_metas = X_transformed[X_train_hm["livermeta"] == "metastasis present"]
        meta_index = variable_names.index("livermeta: metastasis present")
        no_meta_index = variable_names.index("livermeta: no metastasis present")

        np.testing.assert_array_equal(
            no_liver_metas[:, no_meta_index], np.ones(no_liver_metas.shape[0])
        )
        np.testing.assert_array_equal(
            liver_metas[:, meta_index], np.ones(liver_metas.shape[0])
        )

        # 3. adrenal metastases.
        no_adrenal = X_transformed[X_train_hm["adrenalmeta"] == "no metastasis present"]
        no_meta_index = variable_names.index("adrenalmeta: no metastasis present")

        np.testing.assert_array_equal(
            no_adrenal[:, no_meta_index], np.ones(no_adrenal.shape[0])
        )
