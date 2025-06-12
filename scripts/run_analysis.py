import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# --- CONFIGURATION ---
# Get the directory where the current script (run_analysis.py) is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the data file, assuming it's one level up from the script's directory
# AND then inside the 'Data' subdirectory
INSURANCE_DATA_FILE = os.path.join(script_dir, '..', 'Data', 'MachineLearningRating_v3.txt')

# --- Helper Functions (keep the rest as is) ---
def load_and_preprocess_data(file_path):
    """
    Loads the insurance data and performs initial preprocessing.
    Assumes a pipe '|' delimited file.
    """
    try:
        # Read the text file assuming '|' delimiter.
        # Use skipinitialspace=True to handle potential spaces after delimiter.
        # Use dtype to specify types where possible to prevent inference issues.
        df = pd.read_csv(file_path, sep='|', skipinitialspace=True, encoding='utf-8')
        print(f"Dataset '{file_path}' loaded successfully.")

        # Clean column names by stripping whitespace
        df.columns = df.columns.str.strip()

        # Handle potential empty first column name if the first '|' is at the very beginning
        # And also potential empty last column if the last '|' is at the very end
        if df.columns[0] == '':
            df = df.iloc[:, 1:] # Drop the first empty column if it exists
            df.columns = df.columns.str.strip() # Re-strip column names

        if df.columns[-1] == '':
            df = df.iloc[:, :-1] # Drop the last empty column if it exists
            df.columns = df.columns.str.strip() # Re-strip column names


        # Clean trailing '|' and empty rows, if any, often resulting from copy-paste
        df = df.dropna(how='all') # Drop rows where all values are NaN (e.g., empty lines)
        df = df.loc[:, df.columns.notna()] # Drop columns where column name is NaN

        print("\nInitial DataFrame Info:")
        df.info()
        print("\nFirst 5 rows of raw data (after initial load):")
        print(df.head())

        # Convert 'TransactionMonth' to datetime
        if 'TransactionMonth' in df.columns:
            # Handle string formatting like 'YYYY-MM-DD HH:MM:SS' or 'YYYY/MM/DD HH:MM:SS AM/PM'
            # The provided data looks like 'YYYY-MM-DD HH:MM:SS'
            df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
            df.dropna(subset=['TransactionMonth'], inplace=True)
            print("\n'TransactionMonth' converted to datetime and invalid rows removed.")
        else:
            print("\n'TransactionMonth' column not found.")

        # Convert 'TotalPremium' and 'TotalClaims' to numeric, handling errors
        for col in ['TotalPremium', 'TotalClaims', 'CustomValueEstimate']:
            if col in df.columns:
                # Replace comma for thousands separator and convert to numeric
                df[col] = df[col].astype(str).str.replace(',', '', regex=False)
                # Attempt to convert to float, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Fill NaN values with 0 if they represent absence of value for calculations
                df[col] = df[col].fillna(0) # Filling with 0 for premium/claims for ratio calculation

                print(f"'{col}' converted to numeric and NaNs filled with 0.")
            else:
                print(f"'{col}' column not found.")


        return df
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please ensure the file is in the correct directory.")
        return None
    except Exception as e:
        print(f"An error occurred during data loading or initial preprocessing: {e}")
        return None

def calculate_loss_ratio(df):
    """Calculates Loss Ratio (TotalClaims / TotalPremium) and handles division by zero."""
    if 'TotalClaims' in df.columns and 'TotalPremium' in df.columns:
        # Avoid division by zero: if TotalPremium is 0, Loss Ratio is 0.
        # If both TotalClaims and TotalPremium are 0, Loss Ratio is 0.
        df['LossRatio'] = np.where(
            df['TotalPremium'] == 0,
            0,
            df['TotalClaims'] / df['TotalPremium']
        )
        print("\n'LossRatio' calculated (TotalClaims / TotalPremium). Handled division by zero.")
    else:
        print("\n'TotalClaims' or 'TotalPremium' not found. Cannot calculate Loss Ratio.")
    return df

def plot_distribution(df, column, title, xlabel, ylabel='Count', kind='hist', bins=20, rotation=0):
    """Plots distribution for numerical or categorical columns."""
    plt.figure(figsize=(10, 6))
    if kind == 'hist':
        sns.histplot(df[column], kde=True, bins=bins)
    elif kind == 'bar':
        # Ensure column exists and is not empty
        if not df[column].empty:
            sns.countplot(x=column, data=df, order=df[column].value_counts().index)
            plt.xticks(rotation=rotation)
        else:
            print(f"Cannot plot bar chart for {column}: column is empty.")
            plt.close() # Close empty figure
            return
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()

def plot_box(df, column, title, ylabel):
    """Plots a box plot for outlier detection."""
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=df[column])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()

# --- Main EDA Function ---
def perform_eda_insurance(df):
    """
    Performs Exploratory Data Analysis on the insurance DataFrame
    based on the provided guiding questions and KPIs.
    """
    print("\n--- Starting Comprehensive EDA for Insurance Data ---")

    # Data Structure & Initial Quality Assessment
    print("\nDataFrame Info (after initial preprocessing):")
    df.info()
    print("\nDescriptive statistics for numerical columns:")
    print(df.describe())
    print("\nMissing values after initial load and conversions:")
    print(df.isnull().sum()) # Re-check missing values after conversion to numeric/datetime

    # Calculate Loss Ratio
    df = calculate_loss_ratio(df)
    if 'LossRatio' in df.columns:
        print("\nDescriptive statistics for Loss Ratio:")
        print(df['LossRatio'].describe())

    # Guiding Questions & Analysis
    # 1. Overall Loss Ratio & variation by Province, VehicleType, Gender
    print("\n--- Loss Ratio Analysis ---")
    if 'LossRatio' in df.columns and 'Province' in df.columns and not df['Province'].empty:
        loss_ratio_by_province = df.groupby('Province')['LossRatio'].mean().sort_values(ascending=False)
        print("\nAverage Loss Ratio by Province:")
        print(loss_ratio_by_province)
        plt.figure(figsize=(12, 7))
        sns.barplot(x=loss_ratio_by_province.index, y=loss_ratio_by_province.values, palette='coolwarm')
        plt.title('Average Loss Ratio by Province (Creative Plot 1)')
        plt.xlabel('Province')
        plt.ylabel('Average Loss Ratio')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.show()
    else:
        print("\nSkipping Loss Ratio by Province analysis: 'Province' or 'LossRatio' column missing or empty.")

    if 'LossRatio' in df.columns and 'VehicleType' in df.columns and not df['VehicleType'].empty:
        loss_ratio_by_vehicle = df.groupby('VehicleType')['LossRatio'].mean().sort_values(ascending=False)
        print("\nAverage Loss Ratio by Vehicle Type:")
        print(loss_ratio_by_vehicle)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=loss_ratio_by_vehicle.index, y=loss_ratio_by_vehicle.values, palette='viridis')
        plt.title('Average Loss Ratio by Vehicle Type')
        plt.xlabel('Vehicle Type')
        plt.ylabel('Average Loss Ratio')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.show()
    else:
        print("\nSkipping Loss Ratio by Vehicle Type analysis: 'VehicleType' or 'LossRatio' column missing or empty.")

    if 'LossRatio' in df.columns and 'Gender' in df.columns and not df['Gender'].empty:
        loss_ratio_by_gender = df.groupby('Gender')['LossRatio'].mean().sort_values(ascending=False)
        print("\nAverage Loss Ratio by Gender:")
        print(loss_ratio_by_gender)
        # Note: 'Not specified' is a common value, so interpret with caution.
        plt.figure(figsize=(8, 5))
        sns.barplot(x=loss_ratio_by_gender.index, y=loss_ratio_by_gender.values, palette='pastel')
        plt.title('Average Loss Ratio by Gender')
        plt.xlabel('Gender')
        plt.ylabel('Average Loss Ratio')
        plt.tight_layout()
        plt.show()
    else:
        print("\nSkipping Loss Ratio by Gender analysis: 'Gender' or 'LossRatio' column missing or empty.")

    # 2. Distributions of key financial variables & outliers
    print("\n--- Distribution & Outlier Analysis of Financial Variables ---")
    for col in ['TotalPremium', 'TotalClaims', 'CustomValueEstimate']:
        if col in df.columns and not df[col].empty:
            plot_distribution(df, col, f'Distribution of {col}', col, kind='hist')
            plot_box(df, col, f'Box Plot of {col} (Outlier Detection)', col)
            print(f"\nDescriptive statistics for {col}:")
            print(df[col].describe())
        else:
            print(f"\nSkipping analysis for '{col}': column missing or empty.")

    # 3. Temporal trends: Claim frequency or severity over the 18-month period
    print("\n--- Temporal Trends ---")
    if 'TransactionMonth' in df.columns and not df['TransactionMonth'].empty:
        df['MonthYear'] = df['TransactionMonth'].dt.to_period('M')
        # Ensure MonthYear has values before grouping
        if not df['MonthYear'].empty:
            monthly_claims = df.groupby('MonthYear')['TotalClaims'].sum()
            monthly_premiums = df.groupby('MonthYear')['TotalPremium'].sum()
            monthly_policies = df.groupby('MonthYear')['PolicyID'].nunique() # Unique policies per month

            plt.figure(figsize=(15, 7))
            plt.plot(monthly_claims.index.astype(str), monthly_claims.values, marker='o', label='Total Claims')
            plt.plot(monthly_premiums.index.astype(str), monthly_premiums.values, marker='x', label='Total Premium')
            plt.title('Monthly Total Claims and Total Premium Over Time (Creative Plot 2)')
            plt.xlabel('Month-Year')
            plt.ylabel('Amount')
            plt.xticks(rotation=45, ha='right')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()

            # Monthly Loss Ratio Trend
            # Calculate mean Loss Ratio if we want to show it for months where premium was 0 too
            monthly_loss_ratio = df.groupby('MonthYear').apply(lambda x: x['TotalClaims'].sum() / x['TotalPremium'].replace(0, np.nan).sum())
            monthly_loss_ratio = monthly_loss_ratio.fillna(0) # Fill NaN from division by zero with 0 or a more appropriate value

            plt.figure(figsize=(15, 7))
            plt.plot(monthly_loss_ratio.index.astype(str), monthly_loss_ratio.values, marker='s', color='purple', label='Monthly Loss Ratio')
            plt.title('Monthly Loss Ratio Trend Over Time')
            plt.xlabel('Month-Year')
            plt.ylabel('Loss Ratio')
            plt.xticks(rotation=45, ha='right')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()
        else:
            print("\nSkipping temporal trends: 'MonthYear' column is empty after conversion.")
    else:
        print("\nSkipping temporal trends: 'TransactionMonth' column missing or empty.")

    # 4. Vehicle makes/models associated with highest and lowest claim amounts
    print("\n--- Vehicle Make/Model Claims Analysis ---")
    if all(col in df.columns for col in ['make', 'Model', 'TotalClaims']) and not df[['make', 'Model', 'TotalClaims']].empty:
        # Group by make and model, sum claims
        claims_by_vehicle = df.groupby(['make', 'Model'])['TotalClaims'].sum().sort_values(ascending=False)
        print("\nTop 10 Vehicle Make/Models by Total Claims:")
        print(claims_by_vehicle.head(10))
        print("\nBottom 10 Vehicle Make/Models by Total Claims (excluding 0 claims for better insight):")
        # Filter for models that actually had claims to see lowest non-zero claims
        print(claims_by_vehicle[claims_by_vehicle > 0].tail(10))

        # Visualize top 10 makes by total claims (creative plot 3)
        claims_by_make = df.groupby('make')['TotalClaims'].sum().sort_values(ascending=False)
        top_10_makes = claims_by_make.head(10)
        if not top_10_makes.empty:
            plt.figure(figsize=(12, 7))
            sns.barplot(x=top_10_makes.index, y=top_10_makes.values, palette='magma')
            plt.title('Top 10 Vehicle Makes by Total Claim Amount (Creative Plot 3)')
            plt.xlabel('Vehicle Make')
            plt.ylabel('Total Claim Amount')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.75)
            plt.tight_layout()
            plt.show()
        else:
            print("\nNot enough data to plot top 10 vehicle makes by claims.")
    else:
        print("\nSkipping Vehicle Make/Model Claims analysis: Required columns missing or empty.")


    # 5. Correlations between monthly changes TotalPremium and TotalClaims as a function of ZipCode
    print("\n--- Correlation Analysis: Monthly Changes by PostalCode ---")
    if all(col in df.columns for col in ['TransactionMonth', 'TotalPremium', 'TotalClaims', 'PostalCode']) and not df[['TransactionMonth', 'TotalPremium', 'TotalClaims', 'PostalCode']].empty:
        # Aggregate monthly premium and claims per postal code
        monthly_zip_data = df.groupby(['PostalCode', 'TransactionMonth']).agg(
            MonthlyPremium=('TotalPremium', 'sum'),
            MonthlyClaims=('TotalClaims', 'sum')
        ).reset_index()

        # Calculate month-over-month changes
        monthly_zip_data['PremiumChange'] = monthly_zip_data.groupby('PostalCode')['MonthlyPremium'].diff()
        monthly_zip_data['ClaimsChange'] = monthly_zip_data.groupby('PostalCode')['MonthlyClaims'].diff()

        # Calculate correlations for each zip code (only for those with enough data points for change)
        correlations = monthly_zip_data.groupby('PostalCode').apply(
            lambda x: x['PremiumChange'].corr(x['ClaimsChange']) if x[['PremiumChange', 'ClaimsChange']].dropna().shape[0] > 1 else np.nan
        ).dropna().sort_values(ascending=False)

        if not correlations.empty:
            print("\nCorrelation between Monthly Premium Change and Monthly Claims Change by PostalCode (Top 10):")
            print(correlations.head(10)) # Top 10 positive correlations
            print("\nCorrelation between Monthly Premium Change and Monthly Claims Change by PostalCode (Bottom 10):")
            print(correlations.tail(10)) # Top 10 negative correlations (or weakest positive)
        else:
            print("No significant monthly premium/claims change data for correlation analysis by PostalCode.")
    else:
        print("\nSkipping Correlation Analysis by PostalCode: Required columns missing or empty.")


    # 6. Trends Over Geography: Compare the change in insurance cover type, premium, auto make, etc.
    print("\n--- Geographic Trends (Province/PostalCode) ---")
    if 'Province' in df.columns and 'CoverType' in df.columns and not df[['Province', 'CoverType']].empty:
        cover_type_by_province = df.groupby('Province')['CoverType'].value_counts(normalize=True).unstack(fill_value=0)
        print("\nProportion of Cover Types by Province (Top 5 rows):")
        print(cover_type_by_province.head())

        if cover_type_by_province.shape[0] <= 10 and cover_type_by_province.shape[1] <= 10:
             cover_type_by_province.plot(kind='bar', stacked=True, figsize=(12, 7))
             plt.title('Proportion of Cover Types by Province')
             plt.xlabel('Province')
             plt.ylabel('Proportion')
             plt.xticks(rotation=45, ha='right')
             plt.legend(title='Cover Type', bbox_to_anchor=(1.05, 1), loc='upper left')
             plt.tight_layout()
             plt.show()
        else:
             print("\nConsider a heatmap for 'Proportion of Cover Types by Province' due to many categories.")
             plt.figure(figsize=(15, 8))
             sns.heatmap(cover_type_by_province, cmap='YlGnBu', annot=False, fmt=".2f")
             plt.title('Proportion of Cover Types by Province (Heatmap)')
             plt.xlabel('Cover Type')
             plt.ylabel('Province')
             plt.tight_layout()
             plt.show()
    else:
        print("\nSkipping Geographic Trends (CoverType by Province): Required columns missing or empty.")

    print("\n--- Comprehensive EDA Complete ---")
    return df

# --- Main Execution Block ---
def main():
    """
    Main function to run the entire analysis pipeline.
    """
    print("--- Project Pipeline Starting ---")

    # Task 1.2: Data Loading and EDA for Insurance Data
    df_insurance = load_and_preprocess_data(INSURANCE_DATA_FILE)
    if df_insurance is None:
        print("Exiting due to data loading/preprocessing error.")
        return

    df_insurance = perform_eda_insurance(df_insurance)

    # --- Outline for future tasks (remain conceptual as they are not fully developed here) ---
    print("\n--- Starting Task 2: Sentiment Analysis (Conceptual Outline) ---")
    print("This task is designed for text-based data (e.g., financial news headlines).")
    print("It is not directly applicable to the insurance dataset unless text fields (e.g., claim descriptions) are available.")
    print("\n--- Task 2: Sentiment Analysis Outline Complete ---")

    print("\n--- Starting Task 3: Correlation Analysis (Conceptual Outline) ---")
    print("This task typically involves correlating internal data with external factors (e.g., market trends).")
    print("For insurance, this might involve correlating claims/premiums with economic indicators, weather data, etc.")
    print("It requires external data that is not part of this dataset.")
    print("\n--- Task 3: Correlation Analysis Outline Complete ---")

    print("\n--- Starting Task 4: Investment Strategies & Recommendations (Conceptual Outline) ---")
    print("This task is conceptual and relies on insights from previous analysis.")
    print("For insurance, strategies might involve optimizing premiums, identifying high-risk segments, or improving claims processes based on EDA findings.")
    print("\n--- Task 4: Investment Strategies & Recommendations Outline Complete ---")

    print("\n--- Project Pipeline Finished ---")


if __name__ == "__main__":
    main()