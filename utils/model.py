import pandas as pd
from typing import Tuple, List


def test_train_split(pdf_mvp: pd.DataFrame, test_start: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the input DataFrame into train and test sets based on the given test start year.

    Args:
        pdf_mvp: Player data with a 'year' column.
        test_start: Year to use as the start of the test set.

    Returns:
        A tuple of (train_df, test_df)
    """
    train_df = pdf_mvp[pdf_mvp["year"] < test_start]
    test_df = pdf_mvp[pdf_mvp["year"] == test_start]
    return train_df, test_df


def define_features(pdf_mvp: pd.DataFrame, features: List[str]) -> List[str]:
    """
    Expands a feature list to include one-hot encoded subposition or position features.

    Args:
        pdf_mvp: DataFrame that contains one-hot encoded 'subpos_' and 'pos_' columns.
        features: List of desired feature names, which may include 'subpos' or 'pos'.

    Returns:
        A new list of features with 'subpos' or 'pos' replaced by their encoded columns.
    """
    updated_features = features

    if "subpos" in updated_features:
        subpos_features = [col for col in pdf_mvp.columns if col.startswith("subpos_")]
        updated_features.remove("subpos")
        updated_features.extend(subpos_features)

    if "pos" in updated_features:
        pos_features = [col for col in pdf_mvp.columns if col.startswith("pos_")]
        updated_features.remove("pos")
        updated_features.extend(pos_features)

    return updated_features
