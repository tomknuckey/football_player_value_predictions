from typing import List
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from typing import List

def plot_decision_tree_importance(regressor: XGBRegressor, features: List[str]) -> None:
    """
    Plots feature importance for an XGBoost regressor.

    Args:
        regressor: Trained XGBRegressor model.
        features: List of feature names corresponding to model input features.
    """

    importances = regressor.feature_importances_

    # Pair feature names with their importances
    sorted_features = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)

    # Handle empty or all-zero importances gracefully
    if not sorted_features or all(imp == 0 for _, imp in sorted_features):
        print("Warning: All feature importances are zero or no features found.")
        return

    sorted_names, sorted_importances = zip(*sorted_features)

    plt.figure(figsize=(12, 8))
    plt.bar(sorted_names, sorted_importances)
    plt.ylabel("Feature Importance", fontsize=10)
    plt.title("Decision Tree Feature Importance", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show()



def plot_player_value_trends(
    train_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    player_ids: List[int],
    start_year: int = 2015,
    boundary_year: float = 2022.5,
    boundary_label: str = "2022/2023 boundary",
) -> None:
    """
    Plots predicted and historical market values over time for selected players.

    Args:
        train_df: Historical data containing actual market values with columns
                  ["player_id", "year", "age", "market_value_in_million_eur", "name"].
        merged_df: Predicted data with similar columns but "predicted_value" instead of actual.
        player_ids: List of player IDs to filter and plot.
        start_year: Minimum year to include in the plot (default 2015).
        boundary_year: Year at which to add a vertical boundary line (default 2022.5).
        boundary_label: Text label for the boundary line.
    """

    # Prepare historical data, rename target column to match predicted values
    historical_df = train_df[["player_id", "year", "age", "market_value_in_million_eur", "name"]].rename(
        columns={"market_value_in_million_eur": "predicted_value"}
    )

    # Combine actual and predicted data
    combined_data = pd.concat([historical_df, merged_df], ignore_index=True)

    # Filter for selected players and years
    filtered_data = combined_data.query("player_id in @player_ids and year >= @start_year")

    # Create line plot
    fig = px.line(
        filtered_data,
        x="year",
        y="predicted_value",
        color="name",
        title="Predicted Market Values for Selected Players",
        hover_data=["age"],
    )

    # Add vertical boundary line to distinguish actual from predicted
    fig.add_vline(
        x=boundary_year,
        line_dash="dash",
        line_color="red",
        annotation_text=boundary_label,
        annotation_position="top right",
    )

    fig.show()

def save_output_tables(pdf):

    pdf_output_header = pdf[["model_output_id", "model_run_date", "time_taken_seconds", "features_used", "model_type", "split_year"]].drop_duplicates(subset=["model_output_id"])
    pdf_output_detail = pdf[["player_id", "year", "age", "predicted_value", "name", "model_output_id"]]

    # Append header-level output
    pdf_output_header.to_csv("data/output/header_output.csv", mode="a", index=False, header=True)

    # Append detailed player-level output
    pdf_output_detail.to_csv("data/output/detail_output.csv", mode="a", index=False, header=True)
