test_start = 2023

features = [
    "value_last_year",
    "age_last_year",
    "pos",
    "subpos",
    "contract_years_left",
    "team_ppg",
    "team_goal_difference",
    "team_goals_scored",
    "team_goals_conceded",
    "games_played",
    "total_minutes",
    "goals",
    "assists",
    "goal_contributions",
    "goals_per_90",
    "assists_per_90",
    "contrib_per_90",
]

target = "market_value_in_million_eur"

filter_out_synthetic = False # True if we only want real data