import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from pandas.api.types import is_numeric_dtype

def plot_k_distance_graph(data, k):
    """
    Plot the k-distance graph for a given dataset.
    
    Parameters:
    - data: array-like, shape (n_samples, n_features)
      The input data for clustering.
    - k: int
      The number of nearest neighbors to consider.
    """
    # Fit the Nearest Neighbors model
    nbrs = NearestNeighbors(n_neighbors=k).fit(data)
    
    # Compute the distances to the k-th nearest neighbor for each point
    distances, indices = nbrs.kneighbors(data)
    
    # Take the distances to the k-th nearest neighbor
    k_distances = distances[:, k-1]
    
    # Sort the distances in ascending order
    k_distances = np.sort(k_distances)
    
    # Plot the k-distance graph
    plt.figure(figsize=(10, 6))
    plt.plot(k_distances)
    plt.title(f'K-Distance Graph (k = {k})')
    plt.xlabel('Data Points (sorted)')
    plt.ylabel(f'Distance to {k}-th Nearest Neighbor')
    plt.grid(True)
    plt.show()


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

def pick_first_row_every_8(array):
    """
    Selects the first row from every group of 8 rows in a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: A DataFrame with the first row from every group of 8 rows.
    """
    return array#[::8]

def split_dataset(X, y, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets.

    Parameters:
    X (np.ndarray or pd.DataFrame): Features.
    y (np.ndarray or pd.Series): Labels.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.

    Returns:
    tuple: X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def detect_outliers_per_column_and_save_plots(df, method="isolation_forest", contamination=0.05, output_dir="outlier_plots"):
    """
    Detects outliers for each column in the DataFrame and saves scatter plots for visualization.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        method (str): Outlier detection method ('isolation_forest', 'lof', 'elliptic_envelope').
        contamination (float): The proportion of outliers in the data.
        output_dir (str): Directory where plots will be saved.
        
    Returns:
        dict: A dictionary where keys are column names and values are DataFrames with outliers removed for that column.
    """
    import os

    # Create the output directory if it doesn't exist
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    # Initialize a dictionary to store cleaned DataFrames
    cleaned_dfs = {}
    
    # Iterate through each column
    for column in df.columns:

        if df[column].dtype == bool:
            print(f"Skipping non-numeric column: {column}")
            continue
        
        data = df[[column]].values  # Extract single column as 2D array
        
        # Select the method
        if method == "isolation_forest":
            model = IsolationForest(contamination=contamination, random_state=42)
        elif method == "lof":
            model = LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=True)
        elif method == "elliptic_envelope":
            model = EllipticEnvelope(contamination=contamination, support_fraction=1.0, random_state=42)
        else:
            raise ValueError("Invalid method. Choose from 'isolation_forest', 'lof', or 'elliptic_envelope'.")
        
        # Fit the model and predict outliers
        if method == "lof":
            model.fit(data)
            y_pred = model.predict(data)
        else:
            model.fit(data)
            y_pred = model.predict(data)
        
        # Mark inliers (1) and outliers (-1)
        df["Outlier"] = y_pred
        
        if output_dir is not None:
            # Save the scatter plot
            plt.figure(figsize=(10, 6))
            plt.scatter(df[df["Outlier"] == 1][column], 
                        np.random.normal(0, 0.02, len(df[df["Outlier"] == 1])), 
                        color="blue", label="Inliers", alpha=0.7)
            plt.scatter(df[df["Outlier"] == -1][column], 
                        np.random.normal(0, 0.02, len(df[df["Outlier"] == -1])), 
                        color="red", label="Outliers", alpha=0.7)
            plt.title(f"Outlier Detection in {column} using LOF")
            plt.xlabel(column)
            plt.ylabel("Density (random jitter)")
            plt.legend()
            plt.grid(True)
            
            # Save the plot
            plot_path = os.path.join(output_dir, f"{column}_outliers.png")
            plt.savefig(plot_path)
            plt.close()
        
        # Save the cleaned DataFrame for this column
        cleaned_dfs[column] = df[df["Outlier"] == 1].drop(columns=["Outlier"])
        df = df.drop(columns=["Outlier"])
    
    return cleaned_dfs
