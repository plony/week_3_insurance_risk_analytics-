# src/models.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib # For saving/loading models
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from .utils import clean_column_names, convert_to_datetime, calculate_loss_ratio, RAW_DATA_PATH, PROCESSED_DATA_PATH

def prepare_features_for_modeling(df):
    """
    Selects and prepares features for the ML model.
    Handles categorical and numerical features.
    """
    # Define target and features
    target_col = 'total_claims' # Or 'total_premium' for premium prediction (updated name)
    # Features identified from project brief and EDA
    # IMPORTANT: Use the cleaned column names here!
    numerical_features = [
        'custom_value_estimate', 'capital_outstanding', 'sum_insured',
        'term_frequency', 'calculated_premium_per_term',
        'cylinders', 'cubic_capacity', 'kilowatts',
        'number_of_doors', 'registration_year', 'number_of_vehicles_in_fleet'
    ]
    categorical_features = [
        'is_vat_registered', 'citizenship', 'legaltype', 'title', 'language',
        'bank', 'account_type', 'marital_status', 'gender', 'province',
        'postal_code', 'main_cresta_zone', 'sub_cresta_zone', 'itemtype',
        'vehicle_type', 'make', 'model', 'body_type',
        'alarm_immobiliser', 'tracking_device', 'new_vehicle', 'written_off',
        'rebuilt_vehicle', 'converted_vehicle', 'cross_border',
        'cover_category', 'cover_type', 'cover_group', 'section', 'product',
        'statutory_class', 'statutory_risk_type'
    ]

    # Filter available features based on DataFrame columns
    numerical_features = [col for col in numerical_features if col in df.columns]
    categorical_features = [col for col in categorical_features if col in df.columns]

    # Handle high cardinality for postal_code and model: group or drop
    # For now, let's keep them and let OneHotEncoder handle, but be aware of memory/sparsity
    # A more advanced approach would be target encoding or binning for high cardinality

    # Define preprocessor for the ML model (for overall premium prediction)
    # This prepares data for models that can't handle categorical directly
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough' # Keep other columns not transformed
    )

    # Filter out rows where the target is NaN (or fill it appropriately earlier)
    df_model = df.dropna(subset=[target_col]).copy()

    # Drop ID columns, datetime, and engineered loss_ratio (if not target) from features
    X = df_model.drop(columns=[target_col, 'transaction_month', 'underwritten_cover_id', 'policy_id', 'country', 'loss_ratio'], errors='ignore')
    y = df_model[target_col]

    return X, y, preprocessor, numerical_features, categorical_features

def train_linear_regression_per_zipcode(df, target_col='total_claims'):
    """
    Fits a linear regression model for each zipcode to predict total claims.
    Returns a dictionary of models.
    """
    zipcode_models = {}
    unique_zipcodes = df['postal_code'].dropna().unique()

    # Define features for the zipcode-specific models (simpler set)
    # These should be numerical features highly correlated with claims
    # IMPORTANT: Use the cleaned column names here!
    features_for_zipcode_model = [
        'custom_value_estimate', 'sum_insured', 'kilowatts', 'cylinders', 'cubic_capacity',
        'capital_outstanding', 'calculated_premium_per_term', 'number_of_vehicles_in_fleet'
    ]

    print(f"\n--- Training Linear Regression Models per Zipcode for {target_col} ---")
    for zipcode in unique_zipcodes:
        zip_df = df[df['postal_code'] == zipcode].copy()
        zip_df = zip_df.dropna(subset=[target_col])

        # Filter features to only those available in the subset and numeric
        current_features = [f for f in features_for_zipcode_model if f in zip_df.columns and pd.api.types.is_numeric_dtype(zip_df[f])]

        if len(zip_df) > 1 and len(current_features) > 0: # Need at least 2 samples for regression
            X_zip = zip_df[current_features]
            y_zip = zip_df[target_col]

            # Drop rows with NaN in X or y for this specific zipcode
            clean_zip_df = pd.concat([X_zip, y_zip], axis=1).dropna()
            if len(clean_zip_df) < 2:
                print(f"Skipping zipcode {zipcode}: Not enough clean data for regression ({len(clean_zip_df)} samples).")
                continue

            X_zip = clean_zip_df[current_features]
            y_zip = clean_zip_df[target_col]

            model = LinearRegression()
            try:
                model.fit(X_zip, y_zip)
                zipcode_models[zipcode] = model
                y_pred = model.predict(X_zip)
                rmse = np.sqrt(mean_squared_error(y_zip, y_pred))
                r2 = r2_score(y_zip, y_pred)
                print(f"  Zipcode {zipcode}: Model trained. R2={r2:.4f}, RMSE={rmse:.2f} (N={len(y_zip)})")
            except Exception as e:
                print(f"  Error training model for zipcode {zipcode}: {e}")
        else:
            print(f"Skipping zipcode {zipcode}: Not enough data or features for regression.")

    return zipcode_models

def train_and_evaluate_ml_model(X, y, preprocessor, model_type='RandomForest', test_size=0.2, random_state=42):
    """
    Trains and evaluates a machine learning model for optimal premium prediction.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if model_type == 'RandomForest':
        model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    elif model_type == 'LinearRegression':
        model = LinearRegression()
    else:
        raise ValueError(f"Model type '{model_type}' not supported.")

    # Create a pipeline that first preprocesses, then trains the model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])

    print(f"\n--- Training {model_type} Model ---")
    pipeline.fit(X_train, y_train)
    print(f"{model_type} Model Trained.")

    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- {model_type} Model Evaluation ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    return pipeline, X_train, y_train # Return X_train for feature importance if needed

def get_feature_importances(pipeline, numerical_features, categorical_features):
    """
    Extracts and reports feature importances from a tree-based model in the pipeline.
    Assumes the regressor step is a tree-based model (e.g., RandomForestRegressor).
    """
    if not hasattr(pipeline.named_steps['regressor'], 'feature_importances_'):
        print("Regressor does not have 'feature_importances_'. Cannot report.")
        return pd.DataFrame()

    # Get feature names after one-hot encoding
    preprocessor_output_features = []
    for name, transformer, features in pipeline.named_steps['preprocessor'].transformers_:
        if name == 'num':
            preprocessor_output_features.extend(features)
        elif name == 'cat':
            if hasattr(transformer, 'get_feature_names_out'):
                preprocessor_output_features.extend(transformer.get_feature_names_out(features))
            else: # Fallback for older scikit-learn versions
                preprocessor_output_features.extend(features)
        elif name == 'remainder' and transformer != 'passthrough':
            pass # Not expecting complex remainder for now

    all_feature_names = numerical_features + list(pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features))


    importances = pipeline.named_steps['regressor'].feature_importances_
    if len(importances) != len(all_feature_names):
        print(f"Warning: Mismatch between importance array ({len(importances)}) and feature names list ({len(all_feature_names)}).")
        min_len = min(len(importances), len(all_feature_names))
        feature_importances_df = pd.DataFrame({'feature': all_feature_names[:min_len], 'importance': importances[:min_len]})
    else:
        feature_importances_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances})


    feature_importances_df = feature_importances_df.sort_values(by='importance', ascending=False)

    print("\n--- Top Feature Importances ---")
    print(feature_importances_df.head(20)) # Print top 20 features

    return feature_importances_df

def save_model(model, model_name='premium_predictor_model.pkl', path=os.path.join(MODELS_DIR, 'premium_predictor_model.pkl')):
    """
    Saves a trained model to a file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(path=os.path.join(MODELS_DIR, 'premium_predictor_model.pkl')):
    """
    Loads a trained model from a file.
    """
    try:
        model = joblib.load(path)
        print(f"Model loaded from {path}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {path}.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None