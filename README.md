# Football Value Predictor

This uses varies Machine Learning models to predict player values over time  

## Data

https://www.kaggle.com/datasets/davidcariboo/player-scores



## Models

### Decision Tree Traditional 
This does a test train split and predicts 

### Decision Tree Forecasting

This splits the data into test and train based off year.
A decision tree is modelled based off the players value within the last two years of the training set.
Other features are included.
This decision tree is then applied using the training data to predict the first year of the test data.
This is then applied iteratively to predict players value over time.



### Regression

This splits the data into test and train based off year.
A regression is modelled based off the players value within the last two years of the training set.
Other features are included.
This model is then applied using the training data to predict the first year of the test data.
This is then applied iteratively to predict players value over time.