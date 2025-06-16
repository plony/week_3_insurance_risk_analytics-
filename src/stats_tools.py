# src/stats_tools.py
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np

def perform_t_test(group1_data, group2_data, alpha=0.05, equal_var=True, alternative='two-sided'):
    """
    Performs a Student's t-test or Welch's t-test for independent samples.
    """
    t_stat, p_value = stats.ttest_ind(group1_data.dropna(), group2_data.dropna(), equal_var=equal_var, alternative=alternative)
    print(f"\n--- T-Test Results ({alternative}) ---")
    print(f"Group 1 N: {len(group1_data.dropna())}, Mean: {group1_data.mean():.4f}, Std: {group1_data.std():.4f}")
    print(f"Group 2 N: {len(group2_data.dropna())}, Mean: {group2_data.mean():.4f}, Std: {group2_data.std():.4f}")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    if p_value < alpha:
        print(f"Result: Reject the null hypothesis (p < {alpha}). There is a significant difference.")
    else:
        print(f"Result: Fail to reject the null hypothesis (p >= {alpha}). There is no significant difference.")
    return t_stat, p_value

def perform_anova_test(df, value_col, group_col, alpha=0.05):
    """
    Performs one-way ANOVA test for comparing means across multiple groups.
    """
    # Ensure there are at least 2 groups
    if df[group_col].nunique() < 2:
        print(f"ANOVA requires at least 2 groups in '{group_col}'. Found {df[group_col].nunique()}.")
        return None, None

    # Drop rows with NaN in relevant columns for the model
    df_clean = df[[value_col, group_col]].dropna()

    # Perform ANOVA
    formula = f'{value_col} ~ C({group_col})' # C() indicates categorical
    model = ols(formula, data=df_clean).fit()
    anova_table = sm.stats.anova_lm(model, typ=2) # Type 2 sum of squares

    p_value = anova_table['PR(>F)'][group_col] # P-value for the group effect

    print(f"\n--- ANOVA Test Results for {value_col} by {group_col} ---")
    print(anova_table)
    print(f"P-value: {p_value:.4f}")
    if p_value < alpha:
        print(f"Result: Reject the null hypothesis (p < {alpha}). There is a significant difference across groups.")
    else:
        print(f"Result: Fail to reject the null hypothesis (p >= {alpha}). There is no significant difference across groups.")
    return anova_table, p_value

def perform_chi2_test(df, col1, col2, alpha=0.05):
    """
    Performs a Chi-squared test for independence between two categorical variables.
    """
    contingency_table = pd.crosstab(df[col1], df[col2])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    print(f"\n--- Chi-squared Test Results for {col1} vs {col2} ---")
    print("Contingency Table:")
    print(contingency_table)
    print(f"Chi2-statistic: {chi2:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Degrees of Freedom: {dof}")
    if p_value < alpha:
        print(f"Result: Reject the null hypothesis (p < {alpha}). There is a significant association.")
    else:
        print(f"Result: Fail to reject the null hypothesis (p >= {alpha}). There is no significant association.")
    return chi2, p_value