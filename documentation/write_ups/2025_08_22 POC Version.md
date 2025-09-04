We'll be using XG Boost Version - 0.0.10

* **Hyper Parameter Selection:** Optuna  
* **Synthetic Data:** True  
* **Fake Players:** False  
* **Artificial Age Value Cap:** True (80% of last year's value for players ≥ 32)  
* **Min Cap at 0:** True  

**Features:**
- value_last_year
- age_last_year
- pos
- subpos
- contract_years_left
- team_ppg
- team_goal_difference
- team_goals_scored
- team_goals_conceded
- games_played
- total_minutes
- goals
- assists
- goal_contributions
- goals_per_90
- assists_per_90
- contrib_per_90

Currently the model just makes logic for the next year, which is then extrapolated.

RMSE / R squared is for that 1st year, so ensuring sensible results in the long term is more important than 1% of R squared for

## Model Metrics

### Feature Importance

![alt text](image-2.png)

### Shapley Values

![alt text](image-3.png)

We can see that the value last year is the most important feature by a long way, where it has an importance of 0.5, where when it's large it has a positive impact on the model output.

Columns such as goal_contributions, team_ppg, contract_years_left and total_minutes have moderate impact, where the larger they are the more likely they are to have high model outputs.

age_last_year has a small impact, where there have been artificial changes in order to stop players still having high value at very high ages.

Potentially the fact the model only looks at one year causes issues with this where you don't often have big changes in one year.