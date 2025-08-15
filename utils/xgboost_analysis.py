
from typing import List
from sklearn.tree import DecisionTreeRegressor, plot_tree
from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import shap


def plot_basic_decision_tree(
    pdf_train: DataFrame,
    features: List[str],
    target: str,
    max_depth: int = 4,
):
    """
    Trains a DecisionTreeRegressor on the given dataset and plots the resulting tree.
    This helps to visualize how a basic decision tree model makes predictions based on the features.

    Args:
        pdf_train (pd.DataFrame): Training dataset containing features and target.
        features (List[str]): List of feature column names.
        target (str): Target column name.
        max_depth (int, optional): Maximum depth of the tree. Defaults to 4.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    """
    # Train the model
    tree_model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    tree_model.fit(pdf_train[features], pdf_train[target])

    # Plot the tree
    plt.figure(figsize=(24, 16))
    plot_tree(
        tree_model,
        feature_names=features,
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.show()




def plot_feature_importance(model, features: List[str], top_n: int = 20) -> None:
    """
    Plots the top N feature importances from a trained tree-based model.

    Args:
        model (BaseEstimator): Trained model with `feature_importances_` attribute.
        features (List[str]): List of feature names.
        top_n (int): Number of top features to display (default 10).
    """
    importances = pd.Series(model.feature_importances_, index=features)
    top_features = importances.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(8, 6))
    top_features.plot(kind='barh')
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.gca().invert_yaxis()
    plt.show()

    return importances.sort_values(ascending=False)



def create_shapley_values_plots(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    features: List[str]
) -> None:
    """
    Generate and display SHAP summary and waterfall plots for a trained XGBoost model.
    Automatically coerces features to numeric.
    """

    # Ensure numeric dtypes
    X_train = X_train[features].apply(pd.to_numeric, errors="coerce")
    X_test = X_test[features].apply(pd.to_numeric, errors="coerce")

    # SHAP summary plot
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_train)
    shap_values = explanation.values
    shap.summary_plot(shap_values, X_train, feature_names=features)

    # Pick the first test sample
    X_test = X_test.sort_values("value_last_year", ascending=False)

    i = 0
    first_sample = X_test.iloc[[i]]  # Keep DataFrame format

    pred = model.predict(first_sample)

    # SHAP waterfall plot for the first prediction
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[i],
            base_values=explanation.base_values[i],
            data=X_test.iloc[i],
            feature_names=features
        )
    )
