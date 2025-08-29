from typing import List
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from typing import List, Optional, Union
import os
import shap


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

def _prepare_combined_data(train_df: pd.DataFrame, merged_df: pd.DataFrame) -> pd.DataFrame:
    """Combine historical and predicted data, ensuring `current_club_name` exists if possible."""
    base_cols = ["player_id", "year", "age", "market_value_in_million_eur", "name"]
    if "current_club_name" in train_df.columns:
        base_cols.append("current_club_name")

    historical_df = train_df[base_cols].rename(
        columns={"market_value_in_million_eur": "predicted_value"}
    )

    combined = pd.concat([historical_df, merged_df], ignore_index=True)
    return combined
def _filter_players(combined: pd.DataFrame, player_ids: List[Union[int, str]], start_year: int, use_names: bool):
    """Filter by player IDs or names."""
    if use_names:
        return combined.query("name in @player_ids and year >= @start_year")
    return combined.query("player_id in @player_ids and year >= @start_year")


def _filter_teams(combined: pd.DataFrame, teams: List[str], start_year: int):
    """Filter by teams if `current_club_name` exists."""
    if "current_club_name" not in combined.columns:
        return combined.query("year >= @start_year")
    return combined.query("current_club_name in @teams and year >= @start_year")


def _filter_top_n(combined: pd.DataFrame, top_year: int, top_n: int, start_year: int, use_names: bool):
    """Select top-N players by predicted value in a given year."""
    top_players = (
        combined[combined["year"] == top_year]
        .sort_values("predicted_value", ascending=False)
        .head(top_n)
    )
    player_ids = top_players["name"].tolist() if use_names else top_players["player_id"].tolist()
    return _filter_players(combined, player_ids, start_year, use_names)


def _plot_line(filtered_data: pd.DataFrame, boundary_year: float, boundary_label: str):
    """Create the Plotly line chart."""
    hover_cols = ["age"]
    if "current_club_name" in filtered_data.columns:
        hover_cols.append("current_club_name")

    fig = px.line(
        filtered_data,
        x="year",
        y="predicted_value",
        color="name",
        title="Predicted Market Values for Selected Players",
        hover_data=hover_cols,
    )

    fig.add_vline(
        x=boundary_year,
        line_dash="dash",
        line_color="red",
        annotation_text=boundary_label,
        annotation_position="top right",
    )
    return fig


def plot_player_value_trends(
    train_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    player_ids: Optional[List[Union[int, str]]] = None,
    teams: Optional[List[str]] = None,
    top_year: Optional[int] = None,
    top_n: Optional[int] = None,
    start_year: int = 2015,
    boundary_year: float = 2022.5,
    boundary_label: str = "2022/2023 boundary",
    use_names: bool = False,
):
    """Main function to plot player market values using different filtering options."""
    
    combined_data = _prepare_combined_data(train_df, merged_df)

    if player_ids is not None:
        filtered_data = _filter_players(combined_data, player_ids, start_year, use_names)
    elif teams is not None:
        filtered_data = _filter_teams(combined_data, teams, start_year)
    elif top_year is not None and top_n is not None:
        filtered_data = _filter_top_n(combined_data, top_year, top_n, start_year, use_names)
    else:
        raise ValueError("You must provide either player_ids, teams, or top_year/top_n.")

    return _plot_line(filtered_data, boundary_year, boundary_label)



def save_output_tables(pdf):
    header_path = "data/output/header_output.csv"
    detail_path = "data/output/detail_output.csv"

    pdf_output_header = pdf[[
        "model_output_id", "model_run_date", "time_taken_seconds",
        "features_used", "model_type", "split_year", "version"
    ]].drop_duplicates(subset=["model_output_id"])

    pdf_output_detail = pdf[[
        "model_output_id", "player_id", "name", "year", "age", "predicted_value", "actual_value" 
    ]]

    # Check if files exist
    header_exists = os.path.isfile(header_path)
    detail_exists = os.path.isfile(detail_path)

    # Append header-level output
    pdf_output_header.to_csv(header_path, mode="a", index=False, header=not header_exists)

    # Append detailed player-level output
    pdf_output_detail.to_csv(detail_path, mode="a", index=False, header=not detail_exists)
