from typing import List, Optional
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.model import define_features, test_train_split
from utils.output_analysis import plot_player_value_trends
from config import features, test_start, target, filter_out_synthetic, filter_out_fake_players


# Load data
model_id = '9172dbf0-e716-4d4b-b097-239cbe425fca'

pdf_output_header = pd.read_csv("data/output/header_output.csv").query("model_output_id == @model_id")
pdf_output_detail = pd.read_csv("data/output/detail_output.csv")

pdf_output_timeframe= pd.read_csv("data/output/detail_full_timeframe.csv").query("model_output_id == @model_id")

# Merge data
pdf_output = pdf_output_header.merge(pdf_output_detail, on="model_output_id")

# Streamlit page setup
st.set_page_config(page_title="Model Output Dashboard", layout="wide")

st.title("📊 Model Predictions vs Actuals")

# Scatter plot
fig = px.scatter(
    pdf_output,
    x="predicted_value",
    y="actual_value",
    hover_data=["name"],
    title="Predicted vs Actual Values",
    labels={"predicted_value": "Predicted Value", "actual_value": "Actual Value"}
)

st.plotly_chart(fig, use_container_width=True)


pdf_mvp = pd.read_csv("data/intermediate/time_series_model_data_prep.csv") 

pdf_clubs = pd.read_csv("data/intermediate/time_series_model_data_prep.csv").sort_values(by=["player_id", "year"]).groupby("player_id").tail(1)[["player_id", "current_club_name"]].reset_index(drop=True)
features = define_features(pdf_mvp, features)

pdf_train, pdf_test = test_train_split(pdf_mvp, test_start)

if filter_out_synthetic:
    print(pdf_mvp.shape)
    pdf_mvp = pdf_mvp[pdf_mvp["synthetic_flag"] == False]
    print(pdf_mvp.shape)

if filter_out_fake_players:
    print(pdf_mvp.shape)
    pdf_mvp = pdf_mvp[pdf_mvp["player_id"] > 0]
    print(pdf_mvp.shape)

years_available = sorted(pdf_output_timeframe["year"].unique())
default_year = 2025 if 2025 in years_available else years_available[-1]
top_year = st.selectbox("Select year for top-N players", years_available, index=years_available.index(default_year))
top_n = st.slider("Select number of top players to plot", min_value=1, max_value=20, value=10)


fig_trends = plot_player_value_trends(
    pdf_train, pdf_output_timeframe,
    top_year=top_year,
    top_n=top_n
)

st.plotly_chart(fig_trends, use_container_width=True)

# Get intersection of players by id in both dataframes
common_players = set(pdf_train["name"]).intersection(set(pdf_output_timeframe["name"]))

# Filter names only for common players
players = pdf_train.loc[pdf_train["name"].isin(common_players), "name"].unique().tolist()
# Streamlit multiselect for interactivity
selected_players = st.multiselect(
    "Select players to view value trends:",
    options=players,
    default=["Declan Rice"]  # pick any default(s) you like
)

pdf_output_timeframe = pdf_output_timeframe.merge(pdf_clubs, on="player_id", how="left")

# Only plot if at least one player is selected
if selected_players:
    fig_players = plot_player_value_trends(
        pdf_train,
        pdf_output_timeframe,
        player_ids=selected_players,
        use_names=True
    )
    st.plotly_chart(fig_players, use_container_width=True)
else:
    st.info("Please select at least one player to display trends.")


# Get all unique teams that exist in both dataframes
teams = list(pdf_mvp["current_club_name"].unique())

# Streamlit multiselect for teams
selected_teams = st.multiselect(
    "Select teams to view player value trends:",
    options=teams,
)

# Only plot if at least one team is selected
if selected_teams:
    fig = plot_player_value_trends(
        pdf_train,
        pdf_output_timeframe,
        teams=selected_teams
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Please select at least one team to display trends.")
