
We'll be using 0.0.10

* Hyper Parameter Selection - Optuna
* Synthetic Data - True
* Fake Players - False
* Artificial Age Value Cap - True 80% of last years value for players >= 32
* Min Cap at 0 - True

Currently the model just makes logic for the next year, which is then extrapolated.

RMSE / R squared is for that 1st year, so ensuring sensible results in the long term is more important than 1% of R squared for example


