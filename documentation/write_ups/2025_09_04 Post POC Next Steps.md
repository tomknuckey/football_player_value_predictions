# Post-POC: Next Steps for Football Value Predictor

The POC achieves decent results, as explained in [2025_08_22 POC Version](2025_08_22%20POC%20Version.md).

There are many areas for improvement before this can be considered production-ready. The **priority** should be the first two points under **Model Improvements**.

---

## Model Improvements

1. **Move Beyond Single-Year Features**
   - Currently, the model only uses the previous year's data (e.g., goals, value) to predict the next year.
   - **Action:** Incorporate multi-year trends and richer time series features.

2. **Refine Value Capping Logic**
   - At present, if a player is over 32, their value is capped at 80% of the previous year.
   - **Action:** Experiment with alternative caps (e.g., 90% for ages 30–31), and consider position-specific caps (e.g., goalkeepers peak later).
   - The goal is to improve the model so that such caps are rarely needed.

3. **Alternative Approaches**
   - Consider removing age from the model and applying manual adjustments post-prediction.

4. **Explore Other Algorithms**
   - Try ensemble methods such as Random Forest.

5. **Account for Inflation**
   - Remove inflation effects during modeling, then add them back in at the end.

6. **Feature Engineering**
   - Add new features, such as injury record.

7. **Expand Dataset**
   - Run the model on all players, not just those in the Premier League.

---

## Analysis

- Improve and tidy the output framework.
- Save Shapley values to enable post-hoc interpretability plots.
- Improve Streamlit app to suggest players to sell / buy 

---

## Refactoring

- Save the last few versions of outputs and implement safeguards.
- Track model versions in GitHub and update the version log.
- Remove unnecessary run history.
- Speed up Streamlit app performance.
- Save models as pickles to avoid retraining on each run.
- Refactor code for more shared functions and modularity.
- Move more configuration into a config file, and track with tools like MLflow.

---

## Deployment

- Test on additional changeover years.
- Implement data checks and monitoring for data quality.

---