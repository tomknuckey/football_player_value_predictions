import pandas as pd
from typing import Tuple, List
import plotly.express as px
from sklearn.metrics import root_mean_squared_error, r2_score

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

def analysis_result(
    current_df: pd.DataFrame,
    y_test: pd.Series,
    year: int,
    target: str,
) -> None:
    """
    Calculate and print RMSE and R² metrics for predictions, 
    then plot a scatter plot of predicted vs actual values.

    Args:
        current_df (pd.DataFrame): DataFrame containing the predictions with a column "predicted_value" 
                                   and other info columns like "name" and "age" for hover data.
        y_test (pd.Series): Actual target values for comparison.
        year (int): The year of the predictions (used for printing).
        target (str): The name of the target column in current_df to plot against predictions.
    
    Returns:
        None
    """
    rmse_val = root_mean_squared_error(y_test, current_df["predicted_value"])
    r2_val = r2_score(y_test, current_df["predicted_value"])
    print(f"{year} RMSE: {rmse_val:.2f}")
    print(f"{year} R²: {r2_val:.3f}")

    fig = px.scatter(current_df, x="predicted_value", y=target, hover_data=["name", "age"])
    fig.show()