# src/data_tools.py
import pandas as pd
import numpy as np
import os
from utils import clean_column_names, convert_to_datetime, calculate_loss_ratio, RAW_DATA_PATH, PROCESSED_DATA_PATH
def preprocess_data(df):
    """
    Performs initial data cleaning and feature engineering steps.
    """
    if df is None:
        return None

    # Step 1: Clean column names
    df = clean_column_names(df)
    print("Columns cleaned.")

    # Step 2: Convert transaction month to datetime
    df = convert_to_datetime(df, 'transaction_month') # Use cleaned column name
    print("Transaction_month converted to datetime.")

    # Step 3: Handle specific 'Not specified' or empty strings to NaN then fill/drop
    # For Gender, MaritalStatus, Citizenship, LegalType, Title, Language, Bank, AccountType
    # Replace 'Not specified' and empty strings with NaN for proper missing value handling
    for col in ['gender', 'marital_status', 'citizenship', 'legaltype', 'title', 'language', 'bank', 'account_type']:
        if col in df.columns:
            df[col] = df[col].replace(['Not specified', ' ', '', ' '], np.nan) # Added ' ' for robustness
    print("Specific 'Not specified' values handled.")

    # Step 4: Calculate Loss Ratio
    df = calculate_loss_ratio(df)
    print("Loss Ratio calculated.")

    # Step 5: Convert numerical columns that might be object types due to mixed data
    # IMPORTANT: Ensure these column names match the *cleaned* names from clean_column_names
    numerical_cols_to_convert = [
        'custom_value_estimate', 'capital_outstanding', 'sum_insured',
        'term_frequency', 'calculated_premium_per_term', 'total_premium', 'total_claims',
        'cylinders', 'cubic_capacity', 'kilowatts', 'number_of_doors', 'registration_year', 'number_of_vehicles_in_fleet'
    ]
    for col in numerical_cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    print("Key numerical columns converted to numeric.")

    # Example: Simple imputation for numerical NaNs (consider more sophisticated methods later)
    # For demonstration, fill with median for some key numericals if they have NaNs
    for col in ['custom_value_estimate', 'capital_outstanding', 'sum_insured', 'kilowatts', 'cubic_capacity']:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Filled NaN in {col} with median: {median_val}")

    # Example: Simple imputation for categorical NaNs (e.g., mode or 'Unknown')
    for col in ['gender', 'marital_status', 'province', 'vehicle_type', 'body_type', 'make', 'model', 'title']:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna('Unknown')
            print(f"Filled NaN in {col} with 'Unknown'.")

    # Drop columns that are entirely empty or not relevant after initial assessment (adjust as needed)
    # You'd typically find these during EDA
    # Example of dropping columns that are mostly empty or derived
    # df.drop(columns=['unrelevant_col_1', 'unrelevant_col_2'], inplace=True, errors='ignore')

    return df

def save_processed_data(df, file_path=PROCESSED_DATA_PATH):
    """
    Saves the processed DataFrame to a Parquet file for efficient storage.
    """
    if df is not None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_parquet(file_path, index=False)
        print(f"Processed data saved to {file_path}")
    else:
        print("No DataFrame to save.")

def load_processed_data(file_path=PROCESSED_DATA_PATH):
    """
    Loads the processed data from a Parquet file.
    """
    try:
        df = pd.read_parquet(file_path)
        print(f"Processed data loaded successfully from {file_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {file_path}. Please run preprocessing first.")
        return None
    except Exception as e:
        print(f"An error occurred while loading processed data: {e}")
        return None