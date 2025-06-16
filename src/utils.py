# src/utils.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration Constants ---
# --- IMPORTANT: UPDATED DATA FILE NAME AND PATH TO MATCH YOUR STRUCTURE ---
RAW_DATA_PATH = os.path.join('..', 'Data', 'MachineLearningRating_v3.txt') # No 'raw' subfolder, directly under 'Data'
PROCESSED_DATA_PATH = os.path.join('Data', 'processed', 'cleaned_data.parquet') # Create a 'processed' subfolder under 'Data'
MODELS_DIR = 'models' # This path (models/...) remains the same

# --- Data Loading and Basic Inspection ---
def load_raw_data(file_path=RAW_DATA_PATH):
    """
    Loads the historical insurance claims data from a text file,
    assuming it's pipe-delimited based on the sample provided.
    """
    try:
        # Use sep='|' to specify pipe as the delimiter.
        # Your previous sample showed pipe-delimited data.
        # If the new .txt file uses a different delimiter (e.g., comma, tab), adjust `sep`.
        # If there's no header row in the actual MachineLearningRating_v3.txt, add header=None.
        df = pd.read_csv(file_path, sep='|', encoding='utf-8')
        print(f"Raw data loaded successfully from {file_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {file_path}. Please ensure it's downloaded and placed correctly.")
        return None
    except Exception as e:
        print(f"An error occurred while loading raw data: {e}")
        return None

def get_data_summary(df):
    """
    Prints basic descriptive statistics and data types for the DataFrame.
    """
    if df is not None:
        print("\n--- Data Info ---")
        df.info()
        print("\n--- Descriptive Statistics for Numerical Columns ---")
        print(df.describe())
        print("\n--- Value Counts for Top Categorical Columns (first 10) ---")
        for col in df.select_dtypes(include=['object', 'category']).columns:
            if df[col].nunique() < 50: # Only show for columns with fewer than 50 unique values
                print(f"\n--- {col} Value Counts ---")
                print(df[col].value_counts())
            elif df[col].nunique() > 50 and df[col].nunique() < 200: # Show top N for slightly larger unique counts
                print(f"\n--- Top 10 {col} Value Counts ---")
                print(df[col].value_counts().head(10))
    else:
        print("No DataFrame to summarize.")

def check_missing_values(df):
    """
    Returns a DataFrame showing missing value counts and percentages.
    """
    if df is not None:
        missing_counts = df.isnull().sum()
        missing_percent = (df.isnull().sum() / len(df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing_counts,
            'Missing Percent (%)': missing_percent
        })
        return missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)
    return None

def clean_column_names(df):
    """
    Cleans column names by stripping whitespace, removing special characters,
    and converting to snake_case.
    """
    if df is not None:
        cols = df.columns
        new_cols = []
        for col in cols:
            new_col = col.strip()
            new_col = new_col.replace(' ', '_').replace('.', '').replace('(', '').replace(')', '').replace('-', '_').lower()
            # Handle specific common issues (if needed based on your actual column names)
            new_col = new_col.replace('isvatregistered', 'is_vat_registered')
            new_col = new_col.replace('maincresta', 'main_cresta')
            new_col = new_col.replace('subcresta', 'sub_cresta')
            new_col = new_col.replace('mmcode', 'mm_code')
            new_col = new_col.replace('vehicletype', 'vehicle_type')
            new_col = new_col.replace('registrationyear', 'registration_year')
            new_col = new_col.replace('cubiccapacity', 'cubic_capacity')
            new_col = new_col.replace('bodytype', 'body_type')
            new_col = new_col.replace('numberofdoors', 'number_of_doors')
            new_col = new_col.replace('vehicleintrodate', 'vehicle_intro_date')
            new_col = new_col.replace('customvalueestimate', 'custom_value_estimate')
            new_col = new_col.replace('alarmimmobiliser', 'alarm_immobiliser')
            new_col = new_col.replace('trackingdevice', 'tracking_device')
            new_col = new_col.replace('capitaloutstanding', 'capital_outstanding')
            new_col = new_col.replace('newvehicle', 'new_vehicle')
            new_col = new_col.replace('writtenoff', 'written_off')
            new_col = new_col.replace('rebuilt', 'rebuilt_vehicle') # Renamed to avoid conflict if 'rebuilt' is also a value
            new_col = new_col.replace('converted', 'converted_vehicle') # Renamed
            new_col = new_col.replace('crossborder', 'cross_border')
            new_col = new_col.replace('numberofvehiclesinfleet', 'number_of_vehicles_in_fleet')
            new_col = new_col.replace('suminsured', 'sum_insured')
            new_col = new_col.replace('termfrequency', 'term_frequency')
            new_col = new_col.replace('calculatedpremiumperterm', 'calculated_premium_per_term')
            new_col = new_col.replace('excessselected', 'excess_selected')
            new_col = new_col.replace('covercategory', 'cover_category')
            new_col = new_col.replace('covertype', 'cover_type')
            new_col = new_col.replace('covergroup', 'cover_group')
            new_col = new_col.replace('statutoryclass', 'statutory_class')
            new_col = new_col.replace('statutoryrisktype', 'statutory_risk_type')
            new_col = new_col.replace('totalpremium', 'total_premium')
            new_col = new_col.replace('totalclaims', 'total_claims')
            new_col = new_col.replace('underwrittencoverid', 'underwritten_cover_id')
            new_col = new_col.replace('policyid', 'policy_id')
            new_col = new_col.replace('transactionmonth', 'transaction_month')
            new_col = new_col.replace('postalcode', 'postal_code')
            new_col = new_col.replace('accounttype', 'account_type')
            new_col = new_col.replace('maritalstatus', 'marital_status')
            new_cols.append(new_col)
        df.columns = new_cols
    return df

def convert_to_datetime(df, column_name='transaction_month'):
    """
    Converts a specified column to datetime objects.
    Assumes format 'YYYY-MM-DD HH:MM:SS'
    """
    if df is not None and column_name in df.columns:
        df[column_name] = pd.to_datetime(df[column_name], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    return df

def calculate_loss_ratio(df):
    """
    Calculates the Loss Ratio (TotalClaims / TotalPremium) and handles division by zero.
    Ensures total_premium and total_claims are numeric before calculation.
    """
    if df is not None and 'total_premium' in df.columns and 'total_claims' in df.columns:
        # Convert to numeric, coercing errors will turn non-numeric to NaN
        df['total_premium'] = pd.to_numeric(df['total_premium'], errors='coerce')
        df['total_claims'] = pd.to_numeric(df['total_claims'], errors='coerce')

        # Fill NaN in TotalPremium/TotalClaims if they resulted from conversion errors, e.g., with 0 or mean
        df['total_premium'] = df['total_premium'].fillna(0)
        df['total_claims'] = df['total_claims'].fillna(0)

        # Avoid division by zero: if TotalPremium is 0, Loss Ratio is 0 or NaN (or specific handling)
        # Using np.where for vectorized operation, more efficient than apply
        df['loss_ratio'] = np.where(df['total_premium'] != 0,
                                     df['total_claims'] / df['total_premium'],
                                     0) # Or np.nan if you prefer NaN for undefined loss ratio
    else:
        print("Cannot calculate Loss Ratio: Missing 'total_premium' or 'total_claims' columns.")
    return df

# --- Visualization Helpers (for EDA) ---
def plot_numerical_distribution(df, column, title='', bins=50):
    """
    Plots the distribution of a numerical column using histplot and boxplot.
    """
    if df is not None and column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        sns.histplot(df[column], kde=True, bins=bins, ax=axes[0])
        axes[0].set_title(f'Distribution of {title or column}', fontsize=14)
        axes[0].set_xlabel(title or column, fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)

        sns.boxplot(x=df[column], ax=axes[1])
        axes[1].set_title(f'Box Plot of {title or column} (Outliers)', fontsize=14)
        axes[1].set_xlabel(title or column, fontsize=12)
        axes[1].grid(axis='x', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()
    else:
        print(f"Cannot plot numerical distribution for {column}. Column not found or not numerical.")

def plot_categorical_distribution(df, column, title='', top_n=None):
    """
    Plots the distribution of a categorical column using a bar chart.
    Can display top_n categories if specified.
    """
    if df is not None and column in df.columns:
        plt.figure(figsize=(10, 6))
        value_counts = df[column].value_counts()
        if top_n and len(value_counts) > top_n:
            value_counts = value_counts.head(top_n)
            sns.barplot(x=value_counts.values, y=value_counts.index)
            plt.title(f'Top {top_n} Categories for {title or column}', fontsize=14)
        else:
            sns.barplot(x=value_counts.values, y=value_counts.index)
            plt.title(f'Distribution of {title or column}', fontsize=14)
        plt.xlabel('Count', fontsize=12)
        plt.ylabel(title or column, fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    else:
        print(f"Cannot plot categorical distribution for {column}. Column not found.")

def plot_bivariate_categorical_numerical(df, category_col, numerical_col, title='', aggregate_func='mean', top_n=None):
    """
    Plots the relationship between a categorical and a numerical column.
    E.g., Average Loss Ratio by Province.
    """
    if df is not None and category_col in df.columns and numerical_col in df.columns and pd.api.types.is_numeric_dtype(df[numerical_col]):
        grouped_data = df.groupby(category_col)[numerical_col].agg(aggregate_func).sort_values(ascending=False)
        if top_n and len(grouped_data) > top_n:
            grouped_data = grouped_data.head(top_n)

        plt.figure(figsize=(12, 7))
        sns.barplot(x=grouped_data.values, y=grouped_data.index)
        plt.title(f'{aggregate_func.capitalize()} of {numerical_col.replace("_", " ").title()} by {category_col.replace("_", " ").title()} {title}', fontsize=16)
        plt.xlabel(f'{aggregate_func.capitalize()} {numerical_col.replace("_", " ").title()}', fontsize=12)
        plt.ylabel(category_col.replace("_", " ").title(), fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    else:
        print(f"Cannot plot bivariate: Check if columns '{category_col}' and '{numerical_col}' exist and '{numerical_col}' is numerical.")

def plot_correlation_matrix(df, numerical_cols=None):
    """
    Plots a heatmap of the correlation matrix for numerical columns.
    """
    if df is not None:
        if numerical_cols is None:
            numerical_df = df.select_dtypes(include=np.number)
        else:
            numerical_df = df[numerical_cols]

        if numerical_df.empty:
            print("No numerical columns to plot correlation matrix.")
            return

        corr_matrix = numerical_df.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Correlation Matrix of Numerical Features', fontsize=16)
        plt.tight_layout()
        plt.show()
    else:
        print("No DataFrame to plot correlation matrix.")