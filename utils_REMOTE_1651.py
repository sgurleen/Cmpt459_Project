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

def keep_only_numeric_values(odata, feature):
    data = odata.copy()
    # Remove underscores and keep only numerical values and decimal points
    data[feature] = data[feature].astype(str).apply(lambda x: re.sub(r'[^0-9.]', '', x))
    # Convert cleaned values to numeric, setting non-numeric (empty strings) to NaN
    data[feature] = pd.to_numeric(data[feature], errors='coerce')
    return data

## Read Data
def read_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

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