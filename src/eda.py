import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def check_data_quality(df):
    """
    Checks for missing values and duplicate rows in the DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        tuple: A tuple containing:
            - pandas.DataFrame: Missing values count and percentage.
            - int: Number of duplicate rows.
    """
    if df is None:
        print("DataFrame is None. Cannot perform data quality check.")
        return None, None

    missing_values = df.isnull().sum()
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Count': missing_values, 'Missing Percentage (%)': missing_percentage})
    missing_df = missing_df[missing_df['Missing Count'] > 0]

    num_duplicates = df.duplicated().sum()

    return missing_df, num_duplicates

def plot_numerical_distributions(df, columns, save_path='reports/figures/'):
    """
    Plots histograms and KDE for specified numerical columns.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        columns (list): List of numerical column names to plot.
        save_path (str): Directory to save the plots.
    """
    if df is None:
        print("DataFrame is None. Cannot plot numerical distributions.")
        return

    os.makedirs(save_path, exist_ok=True)
    print("\nGenerating distributions of numerical features...")
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            plt.figure(figsize=(8, 5))
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f'Distribution of {col}', fontsize=14)
            plt.xlabel(col, fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'{col}_distribution.png'))
            plt.close()
        else:
            print(f"Warning: Column '{col}' not found or not numerical for plotting.")
    print(f"Numerical distribution plots saved to {save_path}")


def plot_categorical_distributions(df, columns, save_path='reports/figures/'):
    """
    Plots bar charts for specified categorical columns.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        columns (list): List of categorical column names to plot.
        save_path (str): Directory to save the plots.
    """
    if df is None:
        print("DataFrame is None. Cannot plot categorical distributions.")
        return

    os.makedirs(save_path, exist_ok=True)
    print("\nGenerating distributions of categorical features...")
    for col in columns:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            plt.figure(figsize=(10, 6))
            sns.countplot(y=df[col], order=df[col].value_counts().index, palette='viridis')
            plt.title(f'Distribution of {col}', fontsize=14)
            plt.xlabel('Count', fontsize=12)
            plt.ylabel(col, fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'{col}_distribution.png'))
            plt.close()
        else:
            print(f"Warning: Column '{col}' not found or not categorical for plotting.")
    print(f"Categorical distribution plots saved to {save_path}")

def calculate_loss_ratios(df, group_by_cols, metric_col='LossRatio'):
    """
    Calculates average loss ratio for specified grouping columns.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        group_by_cols (list): List of columns to group by.
        metric_col (str): The column containing the metric (e.g., 'LossRatio').

    Returns:
        dict: A dictionary where keys are group_by_cols and values are
              pandas.Series of average loss ratios.
    """
    if df is None:
        print("DataFrame is None. Cannot calculate loss ratios.")
        return {}

    results = {}
    if 'TotalClaims' not in df.columns or 'TotalPremium' not in df.columns:
        print("Required columns 'TotalClaims' or 'TotalPremium' not found for Loss Ratio calculation.")
        return results
    
    # Calculate overall Loss Ratio
    overall_loss_ratio = df['TotalClaims'].sum() / df['TotalPremium'].sum() if df['TotalPremium'].sum() != 0 else 0
    results['Overall'] = overall_loss_ratio
    print(f"\nOverall Loss Ratio for the portfolio: {overall_loss_ratio:.2f}")

    for col in group_by_cols:
        if col in df.columns:
            # Recalculate loss ratio for each group explicitly to avoid issues with pre-calculated 'LossRatio' NaNs
            grouped_data = df.groupby(col).agg(
                TotalClaims_Sum=('TotalClaims', 'sum'),
                TotalPremium_Sum=('TotalPremium', 'sum')
            ).reset_index()
            
            # Handle division by zero for group-specific loss ratios
            grouped_data[metric_col] = grouped_data.apply(
                lambda row: row['TotalClaims_Sum'] / row['TotalPremium_Sum'] if row['TotalPremium_Sum'] != 0 else 0, axis=1
            )
            
            # Sort and store the results
            sorted_results = grouped_data.set_index(col)[metric_col].sort_values(ascending=False)
            results[col] = sorted_results
            print(f"\nAverage {metric_col} by {col}:")
            print(sorted_results)
        else:
            print(f"Warning: Column '{col}' not found for loss ratio calculation.")
    return results


def plot_correlation_matrix(df, numerical_cols, save_path='reports/figures/'):
    """
    Plots a heatmap of the correlation matrix for specified numerical columns.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        numerical_cols (list): List of numerical column names for correlation.
        save_path (str): Directory to save the plot.
    """
    if df is None:
        print("DataFrame is None. Cannot plot correlation matrix.")
        return

    os.makedirs(save_path, exist_ok=True)
    print("\nGenerating correlation matrix...")
    corr_df = df[numerical_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Numerical Features', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'correlation_matrix.png'))
    plt.close()
    print(f"Correlation matrix plot saved to {save_path}")


def analyze_postal_code_profitability(df, save_path='reports/figures/'):
    """
    Analyzes and prints profit margins by postal code.
    Also plots aggregated TotalClaims vs TotalPremium by PostalCode.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        save_path (str): Directory to save the plot.
    """
    if df is None:
        print("DataFrame is None. Cannot analyze postal code profitability.")
        return

    os.makedirs(save_path, exist_ok=True)
    print("\nAnalyzing postal code profitability...")
    postal_code_summary = df.groupby('PostalCode').agg(
        TotalPremium=('TotalPremium', 'sum'),
        TotalClaims=('TotalClaims', 'sum')
    ).reset_index()
    postal_code_summary['ProfitMargin'] = postal_code_summary['TotalPremium'] - postal_code_summary['TotalClaims']
    
    # Handle cases where TotalPremium is 0 for a postal code (to avoid inf in LossRatio if not handled earlier)
    postal_code_summary['LossRatio'] = postal_code_summary.apply(
        lambda row: row['TotalClaims'] / row['TotalPremium'] if row['TotalPremium'] != 0 else 0, axis=1
    )

    postal_code_profit_sorted = postal_code_summary.sort_values(by='ProfitMargin', ascending=True)

    print("\nPostal Codes by Profit Margin (Lowest 10 - potential high risk):")
    print(postal_code_profit_sorted[['PostalCode', 'ProfitMargin', 'LossRatio']].head(10))
    print("\nPostal Codes by Profit Margin (Highest 10 - potential low risk):")
    print(postal_code_profit_sorted[['PostalCode', 'ProfitMargin', 'LossRatio']].tail(10))

    # Scatter plot of TotalClaims vs TotalPremium by PostalCode (Aggregated)
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=postal_code_summary, x='TotalPremium', y='TotalClaims',
                    size='ProfitMargin', sizes=(20, 400), alpha=0.6, legend='brief')
    plt.title('Aggregated Total Claims vs. Total Premium by Postal Code', fontsize=16)
    plt.xlabel('Aggregated Total Premium (ZAR)', fontsize=12)
    plt.ylabel('Aggregated Total Claims (ZAR)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'postal_code_profit_scatter.png'))
    plt.close()
    print(f"Postal code profitability scatter plot saved to {save_path}")

def analyze_temporal_trends(df, save_path='reports/figures/'):
    """
    Analyzes and plots temporal trends for TotalClaims and LossRatio.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        save_path (str): Directory to save the plots.
    """
    if df is None:
        print("DataFrame is None. Cannot analyze temporal trends.")
        return

    os.makedirs(save_path, exist_ok=True)
    print("\nAnalyzing temporal trends...")
    monthly_data = df.groupby(df['TransactionMonth'].dt.to_period('M')).agg(
        TotalClaims_Sum=('TotalClaims', 'sum'),
        TotalPremium_Sum=('TotalPremium', 'sum'),
        NumPolicies=('PolicyID', 'nunique') # Number of unique policies active that month
    ).reset_index()
    monthly_data['TransactionMonth'] = monthly_data['TransactionMonth'].dt.to_timestamp()

    # Calculate claim frequency (if a policy has a claim or not)
    # This requires more detailed tracking, but for simplicity here,
    # we can use NumClaims if we assume each unique policy has one observation per month.
    # A more accurate frequency would be: count of policies with claims / total unique policies per month.
    # For now, let's look at total claims sum.

    monthly_data['MonthlyLossRatio'] = monthly_data.apply(
        lambda row: row['TotalClaims_Sum'] / row['TotalPremium_Sum'] if row['TotalPremium_Sum'] != 0 else 0, axis=1
    )
    monthly_data['MonthlyLossRatio'] = monthly_data['MonthlyLossRatio'].replace([np.inf, -np.inf], np.nan).fillna(0) # In case of 0 premium

    # Plot Total Claims Over Time
    plt.figure(figsize=(14, 6))
    sns.lineplot(data=monthly_data, x='TransactionMonth', y='TotalClaims_Sum', marker='o')
    plt.title('Total Claims Over Time', fontsize=16)
    plt.xlabel('Transaction Month', fontsize=12)
    plt.ylabel('Total Claims Sum (ZAR)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'total_claims_over_time.png'))
    plt.close()

    # Plot Monthly Loss Ratio Over Time
    plt.figure(figsize=(14, 6))
    sns.lineplot(data=monthly_data, x='TransactionMonth', y='MonthlyLossRatio', marker='o', color='red')
    plt.title('Monthly Loss Ratio Over Time', fontsize=16)
    plt.xlabel('Transaction Month', fontsize=12)
    plt.ylabel('Monthly Loss Ratio', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'monthly_loss_ratio_over_time.png'))
    plt.close()
    print(f"Temporal trend plots saved to {save_path}")


def analyze_claims_by_make_model(df):
    """
    Analyzes average claim amounts by vehicle make and model.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        tuple: A tuple containing:
            - pandas.Series: Average claims by make (sorted).
            - dict: Average claims by model for top makes.
    """
    if df is None:
        print("DataFrame is None. Cannot analyze claims by make/model.")
        return None, None

    print("\nAnalyzing average claims by vehicle make and model...")
    avg_claims_by_make = df.groupby('Make')['TotalClaims'].mean().sort_values(ascending=False)
    print("\nAverage Claims by Vehicle Make (Top 10):")
    print(avg_claims_by_make.head(10))
    print("\nAverage Claims by Vehicle Make (Bottom 10):")
    print(avg_claims_by_make.tail(10))

    top_5_makes = avg_claims_by_make.head(5).index
    avg_claims_by_model_for_top_makes = {}
    print("\nAverage Claims by Model for Top 5 Makes:")
    for make in top_5_makes:
        avg_claims_by_model = df[df['Make'] == make].groupby('Model')['TotalClaims'].mean().sort_values(ascending=False)
        avg_claims_by_model_for_top_makes[make] = avg_claims_by_model
        print(f"\n--- {make} Models (Top 5) ---")
        print(avg_claims_by_model.head(5))
    return avg_claims_by_make, avg_claims_by_model_for_top_makes


def plot_boxplots_for_outliers(df, numerical_cols, save_path='reports/figures/'):
    """
    Plots box plots for specified numerical columns to detect outliers.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        numerical_cols (list): List of numerical column names to plot box plots for.
        save_path (str): Directory to save the plots.
    """
    if df is None:
        print("DataFrame is None. Cannot plot boxplots for outliers.")
        return

    os.makedirs(save_path, exist_ok=True)
    print("\nGenerating box plots for outlier detection...")
    for col in numerical_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            plt.figure(figsize=(8, 6))
            sns.boxplot(y=df[col].dropna())
            plt.title(f'Box Plot of {col} (Outlier Detection)', fontsize=14)
            plt.ylabel(col, fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'{col}_boxplot_outliers.png'))
            plt.close()
        else:
            print(f"Warning: Column '{col}' not found or not numerical for boxplot.")
    print(f"Box plots for outliers saved to {save_path}")


def create_insightful_plots(df, loss_ratio_by_province, save_path='reports/figures/'):
    """
    Generates 3 creative and insightful plots.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        loss_ratio_by_province (pandas.Series): Pre-calculated loss ratio by province.
        save_path (str): Directory to save the plots.
    """
    if df is None:
        print("DataFrame is None. Cannot create insightful plots.")
        return

    os.makedirs(save_path, exist_ok=True)
    print("\nGenerating creative and insightful visualizations...")

    # Plot 1: Loss Ratio by Province (highlighting risk)
    plt.figure(figsize=(12, 7))
    sns.barplot(x=loss_ratio_by_province.values, y=loss_ratio_by_province.index, palette='coolwarm')
    plt.title('Average Loss Ratio by Province', fontsize=16)
    plt.xlabel('Average Loss Ratio', fontsize=12)
    plt.ylabel('Province', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'loss_ratio_by_province.png'))
    plt.close()

    # Plot 2: Total Claims vs. Custom Value Estimate, faceted by VehicleType
    plt.figure(figsize=(14, 8))
    # Using 'TotalPremium' for size if it makes sense, otherwise remove or use 'TotalClaims'
    sns.scatterplot(data=df, x='CustomValueEstimate', y='TotalClaims', hue='VehicleType', size='TotalPremium', sizes=(50, 500), alpha=0.7)
    plt.title('Total Claims vs. Custom Value Estimate by Vehicle Type', fontsize=16)
    plt.xlabel('Custom Value Estimate (ZAR)', fontsize=12)
    plt.ylabel('Total Claims (ZAR)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Vehicle Type', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'claims_vs_value_by_vehicle_type.png'))
    plt.close()

    # Plot 3: Distribution of 'TotalClaims' with 'Gender' overlay
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='TotalClaims', hue='Gender', kde=True, multiple='stack', bins=50)
    plt.title('Distribution of Total Claims by Gender', fontsize=16)
    plt.xlabel('Total Claims (ZAR)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(title='Gender')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'total_claims_by_gender_distribution.png'))
    plt.close()
    print(f"Creative plots saved to {save_path}")