# src/models.py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # Added for isinstance checks
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor # Added for isinstance checks
from sklearn.linear_model import LogisticRegression # Added for isinstance checks


class ModelEvaluator:
    def __init__(self, model_type='classification'):
        if model_type not in ['classification', 'regression']:
            raise ValueError("model_type must be 'classification' or 'regression'")
        self.model_type = model_type

    def evaluate(self, y_true, y_pred, y_prob=None, model_name="Model"):
        """
        Evaluates model performance based on its type.
        For classification, plots confusion matrix and ROC curve.
        """
        metrics = {}
        print(f"\n--- {model_name} Performance ({self.model_type.capitalize()}) ---")

        if self.model_type == 'classification':
            metrics['Accuracy'] = accuracy_score(y_true, y_pred)
            metrics['Precision'] = precision_score(y_true, y_pred)
            metrics['Recall'] = recall_score(y_true, y_pred)
            metrics['F1-Score'] = f1_score(y_true, y_pred)
            if y_prob is not None:
                metrics['ROC-AUC'] = roc_auc_score(y_true, y_prob)
            else:
                metrics['ROC-AUC'] = np.nan # Or raise error if ROC-AUC is mandatory for classification

            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")

            cm = confusion_matrix(y_true, y_pred)
            metrics['Confusion Matrix'] = cm
            print("\nConfusion Matrix:")
            print(cm)

            # Plot Confusion Matrix
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=axes[0], cmap='Blues')
            axes[0].set_title(f'Confusion Matrix - {model_name}')

            # Plot ROC Curve
            if y_prob is not None:
                RocCurveDisplay.from_predictions(y_true, y_prob, ax=axes[1])
                axes[1].set_title(f'ROC Curve - {model_name}')
                axes[1].plot([0, 1], [0, 1], 'k--', label='Random Guess') # Add diagonal line
                axes[1].legend()
            else:
                axes[1].set_title(f'ROC Curve - {model_name} (Probabilities not provided)')

            plt.tight_layout()
            plt.show() # <--- Crucial for displaying plots

        elif self.model_type == 'regression':
            metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics['R-squared'] = r2_score(y_true, y_pred)

            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")
            
            # Plotting actual vs predicted
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title(f"{model_name}: Actual vs Predicted Values")
            plt.grid(True)
            plt.show() # <--- Crucial for displaying plots

        return metrics


class ModelInterpreter:
    def __init__(self, model, X_transformed_df):
        self.model = model
        self.X_transformed_df = X_transformed_df
        self.explainer = None
        self.shap_values = None

    def _initialize_shap(self):
        if self.explainer is None:
            # For tree-based models, use TreeExplainer
            if hasattr(self.model, 'tree_') or isinstance(self.model, (
                RandomForestClassifier, RandomForestRegressor,
                DecisionTreeClassifier, DecisionTreeRegressor,
                xgboost.XGBClassifier, xgboost.XGBRegressor
            )):
                try:
                    self.explainer = shap.TreeExplainer(self.model)
                except Exception:
                    # Fallback for models not directly supported by TreeExplainer (e.g., if it's wrapped)
                    print("Warning: TreeExplainer failed, falling back to KernelExplainer. This might be slow.")
                    self.explainer = shap.KernelExplainer(self.model.predict, self.X_transformed_df.sample(min(100, len(self.X_transformed_df)), random_state=42))
            else:
                print("Using KernelExplainer. This might be slow for large datasets.")
                self.explainer = shap.KernelExplainer(self.model.predict, self.X_transformed_df.sample(min(100, len(self.X_transformed_df)), random_state=42))

            # Calculate SHAP values
            if isinstance(self.model, (LogisticRegression,)): # For models where predict_proba is preferred for SHAP
                # For binary classification, shap_values will be a list of two arrays. We take the one for the positive class.
                self.shap_values = self.explainer.shap_values(self.X_transformed_df)
                if isinstance(self.shap_values, list) and len(self.shap_values) == 2:
                    self.shap_values = self.shap_values[1] # SHAP values for the positive class (class 1)
            else:
                self.shap_values = self.explainer.shap_values(self.X_transformed_df)

    def plot_feature_importance(self, num_features=10, plot_type='bar'):
        """
        Plots global feature importance using SHAP values.
        plot_type can be 'bar' (mean absolute SHAP) or 'summary' (beeswarm plot).
        """
        self._initialize_shap()

        if self.shap_values is None:
            print("SHAP values could not be calculated. Skipping feature importance plot.")
            return

        plt.figure(figsize=(10, 6))
        if plot_type == 'bar':
            shap.summary_plot(self.shap_values, self.X_transformed_df, plot_type="bar", show=False)
            plt.title(f"Top {num_features} Features by Mean Absolute SHAP Value")
        elif plot_type == 'summary':
            shap.summary_plot(self.shap_values, self.X_transformed_df, max_display=num_features, show=False)
            plt.title(f"SHAP Summary Plot of Top {num_features} Features")
        else:
            raise ValueError("plot_type must be 'bar' or 'summary'.")

        plt.tight_layout()
        plt.show() # <--- Crucial for displaying plots

        # Return top features for textual interpretation
        if isinstance(self.shap_values, np.ndarray):
            # Calculate mean absolute SHAP values
            mean_abs_shap_values = np.abs(self.shap_values).mean(axis=0)
            feature_importance_df = pd.DataFrame({
                'Feature': self.X_transformed_df.columns,
                'SHAP_Importance': mean_abs_shap_values
            })
            top_features_df = feature_importance_df.sort_values(by='SHAP_Importance', ascending=False).head(num_features)
            print("\n--- Top Features by SHAP Importance ---")
            print(top_features_df)
            return top_features_df
        else:
            return None # Or handle other SHAP value types