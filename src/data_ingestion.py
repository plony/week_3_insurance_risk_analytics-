import pandas as pd

def load_data(filepath):
    """
    Loads data from a specified CSV file.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}.")
        return df
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

def preprocess_initial(df):
    """
    Performs initial data preprocessing steps:
    - Converts 'TransactionMonth' and 'VehicleIntroDate' to datetime.
    - Handles potential errors during date conversion.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.DataFrame: The preprocessed DataFrame.
    """
    if df is None:
        return None

    # Convert date columns
    if 'TransactionMonth' in df.columns:
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
    if 'VehicleIntroDate' in df.columns:
        df['VehicleIntroDate'] = pd.to_datetime(df['VehicleIntroDate'], errors='coerce')

    # Handle potential inf/-inf from division by zero for LossRatio later
    # This is anticipatory for EDA calculations.
    df['TotalPremium'] = df['TotalPremium'].replace(0, np.nan) # Replace 0 with NaN for safe division
    df['LossRatio'] = df['TotalClaims'] / df['TotalPremium']
    df['LossRatio'] = df['LossRatio'].replace([np.inf, -np.inf], np.nan)
    # Fill NaN LossRatio where TotalClaims was also 0 (e.g., 0/0) with 0, or other NaNs with median/mean
    # For now, fill with 0 based on previous analysis of "no claim = no loss"
    df['LossRatio'] = df['LossRatio'].fillna(0)
    df['TotalPremium'] = df['TotalPremium'].fillna(0) # Revert NaN back to 0 if preferred for other calcs

    return df