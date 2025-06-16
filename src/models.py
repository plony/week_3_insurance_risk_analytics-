# src/models.py
# models.py

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import shap
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """
    A class to evaluate regression and classification models.
    """
    def __init__(self, model_type='regression'):
        # Validates model_type upon initialization
        if model_type not in ['regression', 'classification']:
            raise ValueError("model_type must be 'regression' or 'classification'")
        self.model_type = model_type

    def evaluate(self, y_true, y_pred, y_prob=None, model_name="Model"):
        """
        Evaluates the model based on its type.
        For regression: RMSE, R-squared.
        For classification: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix.
        
        Parameters:
        - y_true (array-like): True labels or values.
        - y_pred (array-like): Predicted labels or values.
        - y_prob (array-like, optional): Predicted probabilities for the positive class (for classification).
        - model_name (str): Name of the model being evaluated.
        
        Returns:
        - dict: A dictionary containing the evaluation metrics.
        """
        print(f"\n--- {model_name} Performance ({self.model_type.capitalize()}) ---")
        results = {}

        if self.model_type == 'regression':
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            print(f"RMSE: {rmse:.4f}")
            print(f"R-squared: {r2:.4f}")
            results['RMSE'] = rmse
            results['R-squared'] = r2
        elif self.model_type == 'classification':
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0) # zero_division=0 to handle cases where no positive predictions are made
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            cm = confusion_matrix(y_true, y_pred)

            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print("\nConfusion Matrix:")
            print(cm)

            results['Accuracy'] = accuracy
            results['Precision'] = precision
            results['Recall'] = recall
            results['F1-Score'] = f1
            results['Confusion Matrix'] = cm

            if y_prob is not None:
                # Check if there's at least one positive class in y_true for ROC-AUC
                if len(np.unique(y_true)) > 1:
                    try:
                        roc_auc = roc_auc_score(y_true, y_prob)
                        print(f"ROC-AUC: {roc_auc:.4f}")
                        results['ROC-AUC'] = roc_auc
                    except ValueError:
                        print("ROC-AUC could not be calculated (requires positive class in y_true and y_prob).")
                        results['ROC-AUC'] = np.nan
                else:
                    print("ROC-AUC not applicable: only one class present in true labels.")
                    results['ROC-AUC'] = np.nan

        return results

class ModelInterpreter:
    """
    A class for model interpretability using SHAP values.
    """
    def __init__(self, model, X_transformed_df):
        self.model = model
        self.X_transformed_df = X_transformed_df # DataFrame of preprocessed features
        
        # Initialize explainer based on model type (TreeExplainer for tree models, KernelExplainer for others)
        # Check if the model has a 'feature_importances_' attribute or is an XGBoost/RandomForest model
        model_type_str = str(type(model))
        if "XGB" in model_type_str or "RandomForest" in model_type_str:
            self.explainer = shap.TreeExplainer(model)
        else:
            # For linear models, KernelExplainer works but can be slow.
            # Using a background dataset sample for performance.
            if self.X_transformed_df.empty:
                print("Warning: X_transformed_df is empty, KernelExplainer cannot be initialized.")
                self.explainer = None
            else:
                background_data_sample = self.X_transformed_df.sample(min(200, len(self.X_transformed_df)), random_state=42) # Increased sample size for better approximation
                self.explainer = shap.KernelExplainer(model.predict, background_data_sample)

    def plot_feature_importance(self, num_features=10, plot_type='bar'):
        """
        Generates and plots feature importance (SHAP values).
        
        Parameters:
        - num_features (int): Number of top features to display.
        - plot_type (str): Type of SHAP plot ('bar' or 'summary').
        
        Returns:
        - pd.DataFrame: A DataFrame of top features and their SHAP importance.
        """
        if self.X_transformed_df.empty or self.explainer is None:
            print("No data or explainer available for SHAP explanation.")
            return pd.DataFrame()

        # Calculate SHAP values for a subset of the data for performance for non-tree models
        # For tree models, can use more data or even all if feasible.
        sample_size = min(2000, len(self.X_transformed_df)) # Increased sample size for better SHAP plots
        X_sample = self.X_transformed_df.sample(sample_size, random_state=42)
        
        shap_values = self.explainer.shap_values(X_sample)

        # For multi-output models (e.g., LogisticRegression for binary classification), shap_values might be a list.
        # Take the SHAP values for the positive class (class 1)
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values = shap_values[1] # For binary classification, index 1 is the positive class

        # Summarize SHAP values (mean absolute SHAP value for each feature)
        # Ensure shap_values is 2D for mean(0)
        if len(shap_values.shape) > 2: # For potential multi-output explanations not handled by list unpacking
             shap_values = shap_values.reshape(-1, shap_values.shape[-1]) # Flatten to 2D if necessary

        shap_abs_mean = np.abs(shap_values).mean(0)
        
        feature_importance_df = pd.DataFrame({
            'Feature': X_sample.columns,
            'SHAP_Importance': shap_abs_mean
        }).sort_values(by='SHAP_Importance', ascending=False)

        print(f"\n--- Top {num_features} Features by SHAP Importance ---")
        print(feature_importance_df.head(num_features))

        plt.figure(figsize=(12, 7)) # Adjusted figure size for better readability
        if plot_type == 'bar':
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=num_features)
        elif plot_type == 'summary':
            shap.summary_plot(shap_values, X_sample, show=False, max_display=num_features)
        plt.title(f"SHAP Feature Importance ({'Mean Absolute' if plot_type == 'bar' else 'Summary'})")
        plt.tight_layout()
        plt.show()

        return feature_importance_df.head(num_features)