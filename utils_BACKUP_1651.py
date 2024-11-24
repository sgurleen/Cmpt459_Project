import re
import pandas as pd
import numpy as np

def convert_to_months(data, feature):
    
    data[feature] = data[feature].apply(
        lambda value: int(match.group(1)) * 12 + int(match.group(2))
        if isinstance(value, str) and (match := re.match(r"(\d+)\s+Years?\s+and\s+(\d+)\s+Months?", value))
        else None
    )

    return data

def count_numerical_values(data, feature):
    numerical_count = pd.to_numeric(data[feature], errors='coerce').notnull().sum()
    print(f"Number of numerical values in '{feature}': {numerical_count}")
    non_numerical_values = data[~pd.to_numeric(data[feature], errors='coerce').notnull()][feature]
    print(non_numerical_values)
    # return numerical_count

def keep_only_numeric_values(data, feature):

    # Remove underscores and keep only numerical values and decimal points
    data[feature] = pd.to_numeric(data[feature].str.strip("_"), downcast="integer")

    return data

## Read Data
def read_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

<<<<<<< HEAD
def split_and_duplicate_rows(df, column_name):
    df["Type_of_Loan"] = df["Type_of_Loan"].str.replace(" and", "", regex=False)
    df["Type_of_Loan"] = df["Type_of_Loan"].str.split(", ")

    # List of unique loan values
    unique_loan_types = ['Auto Loan', 'Credit-Builder Loan', 'Debt Consolidation Loan', 'Home Equity Loan', 'Mortgage Loan',
                        'Zero Loan', 'Not Specified', 'Payday Loan', 'Personal Loan', 'Student Loan']

    df = df.dropna(subset=['Type_of_Loan'])
    
    for loan_type in unique_loan_types:
        df[loan_type] = df[column_name].apply(lambda x: x.count(loan_type))

    return df

def forward_backward_filling(data, col, id_col):
    # Use groupby to apply forward fill and backward fill within each Customer_ID group
    data[col] = (
        data.groupby(id_col, group_keys=False)[col]
        .apply(lambda group: group.ffill().bfill())
        .reset_index(drop=True)
    )

    return data

def fill_with_group_mode(data, col, id_col):
    def safe_mode(series):
        # Return the first mode if it exists; otherwise, return NaN
        modes = series.mode()
        return modes.iloc[0] if not modes.empty else np.nan

    # Compute the mode for each group
    group_modes = data.groupby(id_col)[col].transform(safe_mode)

    # Replace missing values with the group mode
    # data[col] = data[col].fillna(group_modes)
    data[col] = data.groupby(id_col)[col].transform(safe_mode)

    return data

def print_nan_counts(data):
    nan_counts = data.isna().sum()
    print("Column Name | NaN Count")
    print("-" * 25)
    for col, count in nan_counts.items():
        print(f"{col:<12} | {count}")
=======
## To transform values so that we can see clear outliers for boxplots
def apply_transformation(data, numerical_columns):
    """
    Applies the specified transformation to a pandas Series.
    """
    log_transformed_data = data.copy()  # Create a copy of the dataset
    for col in numerical_columns:
        # Ensure the column has only positive values before applying log
        if (log_transformed_data[col] > 0).all():
            log_transformed_data[col] = np.log1p(log_transformed_data[col])  # Use log1p to handle small values
        else:
            print(f"Skipping column '{col}' as it contains non-positive values.")
    return log_transformed_data
>>>>>>> 9a7e6cf30641a650343f5334d07c0e8f08bbb3b3
