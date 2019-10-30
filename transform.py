from typing import Callable, Iterable, List, Optional, Tuple

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder


def dummy_encode_mutations(
    data_frame: pd.DataFrame, gene_vocabulary: Iterable
) -> pd.DataFrame:
    """
    Dummy encode mutations for each patient.
    """
    # Create an empty data frame first.
    dummy_data_frame = pd.DataFrame(
        0,
        # Rows are patients
        index=data_frame["Patient ID"].unique(),
        # Columns are the number of mutations per gene.
        columns=gene_vocabulary,
    )

    # Fill all columns with the number of occurences of each gene per patient.
    for multi_index, patient_genes in data_frame.groupby(["Patient ID", "Gene"]):
        # Unpack patient_id and the gene from the `MultiIndex`.
        patient_id, gene = multi_index
        # Store number of mutations of this particular gene.
        dummy_data_frame.loc[patient_id][gene] += len(patient_genes)

    return dummy_data_frame


def categorical_columns_to_lower(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all text columns to lower case.
    """
    df = data_frame.copy()
    for column in data_frame.columns:
        # This is a dirty way to check if it is non-numeric, but pandas thinks
        # all the columns are strings.
        try:
            float(data_frame[column].iloc[0])
        except ValueError:
            df[column] = data_frame[column].str.lower()

    return df


def patient_allele_frequencies(
    data_frame: pd.DataFrame,
    gene_vocabulary: Iterable,
    transformation: Callable = lambda x, y: y - x,
    allele_columns: List[str] = ["T0: Allele \nFraction", "T1: Allele Fraction"],
    handle_duplicates="sum",
) -> pd.DataFrame:
    """
    For each patient, calculate allele frequency increase (by default) of
    mutations.

    For each mutation for a given patient, calculate `transformation(f_t0,
    f_t1)` from the allele frequencies measured as t0 and t1.
    """
    allowed_duplicate_actions = ("min", "max", "ignore", "concat", "sum")
    if handle_duplicates.lower() not in allowed_duplicate_actions:
        raise ValueError(
            "Allowed values for handle_duplicates are {}.".format(
                allowed_duplicate_actions
            )
        )

    # The transformation is calculated between two columns (t0 and t1).
    if len(allele_columns) != 2:
        raise ValueError("Allele frequency columns must be precisely two!")

    # The columns that were passed must actually exist.
    column_names = data_frame.columns
    if (
        allele_columns[0] not in column_names
        or allele_columns[1] not in column_names
    ):
        raise KeyError(
            "Column lookup error in `allel_freq_columns` = {}.".format(
                allele_columns
            )
        )

    # There may not be any NA values in the columns.
    if (
        sum(data_frame[allele_columns[0]].isna()) > 0
        or sum(data_frame[allele_columns[1]].isna()) > 0
    ):
        raise ValueError("NA values found in allele frequency columns!")

    # Create an empty data frame first.
    patient_data_frame = pd.DataFrame(
        0.0,
        # Rows are patients
        index=data_frame["Patient ID"].unique(),
        # Columns are the calculated allele frequency differences (unless
        # specified otherwise with `transformation`) per gene mutation.
        columns=gene_vocabulary,
    )

    def first_element(x):
        return x.iloc[0]

    # Default is ignore, taking the first element that is found.
    select_from_duplicates = first_element
    if handle_duplicates == "min":
        select_from_duplicates = np.min
    elif handle_duplicates == "max":
        select_from_duplicates = np.max
    elif handle_duplicates == "sum":
        select_from_duplicates = np.sum

    # Go through all the mutations.
    for group_index, grouped in data_frame.groupby(["Patient ID", "Gene"]):
        patient_id, gene = group_index

        # Extract allele frequencies at time t0 and t1.
        f_t0, f_t1 = (grouped[allele_columns[0]], grouped[allele_columns[1]])

        # Carry out the transformation on the two allele frequencies (by default
        # difference), and store result in the corresponding gene column for the
        # given patient.
        f = transformation(f_t0, f_t1)
        patient_data_frame.loc[patient_id, gene] = select_from_duplicates(f)

    return patient_data_frame


def mutation_train_test_split(
    mutation_data_frame: pd.DataFrame, test_fraction: float = 0.3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the mutation data frame in a train and test set.
    """
    patients = mutation_data_frame["Patient ID"].unique()
    np.random.shuffle(patients)

    # Determine the largest element no longer in the training set.
    cut_off = int(np.ceil(len(patients) * (1 - test_fraction)))
    train_patients = patients[:cut_off]

    # Put records in train/validation set according to "Patient ID".
    train_rows = mutation_data_frame["Patient ID"].isin(train_patients)
    train_data_frame = mutation_data_frame.loc[train_rows]
    test_data_frame = mutation_data_frame[~train_rows]

    return (train_data_frame, test_data_frame)


def get_top_correlated(
    correlations: pd.DataFrame,
    correlation_pvalue: Optional[pd.DataFrame] = None,
    ascending: bool = False,
    top_count: int = 15,
    gene_counts: Optional = None,
) -> pd.DataFrame:
    """
    Get the top correlated genes in ascending (descending) order.
    """
    # Get the maximal cell by:
    # 1) flatting array.
    corr_flat = correlations.values.flatten()

    # 2) and sorting indices.
    if not ascending:
        top_indices = np.argsort(corr_flat)
    else:
        top_indices = np.argsort(corr_flat)

    # 3) Calculating indices back to original dataframe.
    i, j = np.unravel_index(top_indices, correlations.shape)

    # Finally store results in data frame.
    columns = {
        "gene 1": correlations.index[i],
        "gene 2": correlations.index[j],
        "correlation": corr_flat[top_indices],
    }

    if correlation_pvalue is not None:
        pval_flat = correlation_pvalue.values.flatten()
        columns["p-value"] = pval_flat[top_indices]

    df = pd.DataFrame(columns)

    # Add columns with gene counts if passed.
    if gene_counts is not None:
        df["# gene 1"] = gene_counts[correlations.index[i]].values
        df["# gene 2"] = gene_counts[correlations.index[j]].values

    # Remove diagonal elements (i, i).
    diagonal_genes = df["gene 1"] == df["gene 2"]
    df = df[~diagonal_genes]
    # Remove permutations (i, j) ~ (j, i).
    even_indices = df.index % 2 == 0
    df = df[even_indices]

    # Sort and truncate size.
    return df.sort_values([df.columns[2]], ascending=ascending).iloc[:top_count]


def clean_mutation_columns(
    input_data: pd.DataFrame,
    columns_to_number=(
        "T0: Allele \nFraction",
        "T1: Allele Fraction",
        "T0: No. Mutant \nMolecules per mL",
        "T1: No. Mutant \nMolecules per mL",
    ),
) -> pd.DataFrame:
    """
    Convert mutation data to floats and drop missing values.
    """
    clean_data = input_data.copy()
    # 1) Convert to float.
    for column_name in columns_to_number:
        clean_data.loc[:, column_name] = pd.to_numeric(
            input_data[column_name], errors="coerce"
        )

    # 2) Drop rows for which the columns can not be converted.
    return clean_data.dropna(subset=columns_to_number)


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


def get_top_genes(data_frame: pd.DataFrame, thresshold: int = 5) -> pd.DataFrame:
    """
    Thresshold: genes must occur at least this many times.
    """
    # What elements are non-zero?
    non_zero_values = data_frame[data_frame != 0]
    # Pick columns that have at least `LAMBDA` occurences.
    gene_thressholded = non_zero_values.count() >= thresshold
    genes_to_pick = gene_thressholded.index[gene_thressholded]
    frequent_mutations = genes_to_pick.values
    return frequent_mutations
