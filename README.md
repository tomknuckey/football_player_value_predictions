# Football Value Predictor

This uses varies Machine Learning models to predict player values over time, where we use XG Boost.
The outputs are displayed on a streamlit app.  

## Data

The input data is from https://www.kaggle.com/datasets/davidcariboo/player-scores


## How to Use the Streamlit App

The Streamlit dashboard provides an interactive way to explore and visualize player value predictions from the models in this project.

This currently takes results from version 0.0.10 as defined here 
documentation\write_ups\2025_08_22 POC Version.md

Forecast ID = '9172dbf0-e716-4d4b-b097-239cbe425fca'

### 1. **Install Requirements**

Make sure you have all dependencies installed:

```bash
pip install -r requirements.txt
```

### 2. **Prepare Data**

Ensure the required data files are present in the `data/output/` and `data/intermediate/` directories. You should have:
- `header_output.csv`
- `detail_output.csv`
- `detail_full_timeframe.csv`
- `time_series_model_data_prep.csv` (in `data/intermediate/`)

The input data is from https://www.kaggle.com/datasets/davidcariboo/player-scores

### 3. **Run the App**

From the project root directory, start the Streamlit app with:

```bash
streamlit run app.py
```

### 4. **Using the Dashboard**

- **Predicted vs Actual Scatter Plot:**  
  The app displays a scatter plot comparing predicted and actual player values for the selected model run.

- **Top-N Player Value Trends:**  
  Use the sidebar or controls to select the year (`top_year`) and the number of top players (`top_n`) you want to visualize. The app will plot the predicted value trajectories for the top-N players in the selected year.

- **Filtering Options:**  
  The app supports filtering out synthetic or fake players if configured in `config.py`.

### 5. **Customizing the App**

- You can change the model run by editing the `model_id` variable in `app.py`.
- Adjust the features and filtering logic in `config.py` as needed.

### 6. **Troubleshooting**

- If you see missing data or errors, ensure all required CSV files are up to date and in the correct locations.
- For further customization, edit the plotting and data loading logic in `app.py` and `utils/output_analysis.py`.


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