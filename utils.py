import pandas as pd


def get_categorical_columns(data_frame: pd.DataFrame) -> list:
    """
    Convert all text columns to lower case.
    """
    categorical_columns = []
    for column in data_frame.columns:
        # This is a dirty way to check if it is non-numeric, but pandas thinks
        # all the columns are strings.
        try:
            float(data_frame[column].iloc[0])
        except ValueError:
            categorical_columns.append(column)

    return categorical_columns
