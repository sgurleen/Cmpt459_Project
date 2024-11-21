import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import utils

########## Data Preprocessing ##################
def filter_numeric_data(data):

    # Also observed some non-numeric values in Age
    data = utils.keep_only_numeric_values(data, 'Age')
    # Observed in EDA that age has some values < 1 and > 100
    data = data[(data['Age'] > 0) & (data['Age'] < 100)]

    # Remove non-numeric values from Annual Income
    data = utils.keep_only_numeric_values(data, 'Annual_Income')
    data = data[(data['Annual_Income'] > 100) ] # Some values are way too small

    # Remove all underscores from Num of Loan
    data = utils.keep_only_numeric_values(data, 'Num_of_Loan')
    data = utils.keep_only_numeric_values(data, 'Num_of_Delayed_Payment')
    data = data[(data['Num_of_Delayed_Payment'] >= 0) ]
    # Observed some values with only _
    data = data[data['Changed_Credit_Limit'] != '_']

    data = utils.keep_only_numeric_values(data, 'Changed_Credit_Limit')
    data = utils.keep_only_numeric_values(data, 'Outstanding_Debt')
    data = utils.keep_only_numeric_values(data, 'Amount_invested_monthly')
    data = utils.keep_only_numeric_values(data, 'Monthly_Balance')
    data = data[(data['Monthly_Balance'] > 0)]

    data = data[data['Payment_Behaviour'] != '!@9#%8']

    return data


def encode_categorical_to_numerical(data, feature):
    encoded_data = utils.convert_to_months(data, feature)

    return encoded_data

def label_encoding(data, feature1, feature2):
    """
    Encodes categorical labels into integers based on predefined mapping.

    Mapping:
        - "good" -> 2
        - "bad" -> 1
        - "standard" -> 0

    """
    mapping1 = {
        "Good": 2,
        "Bad": 1,
        "Standard": 0
    }

    mapping2 = {
        'Yes': 0,
        'NM': 1,
        'No': 2
    }

    data[feature1] = data[feature1].map(mapping1)
    data[feature2] = data[feature2].map(mapping2)

    return data

def one_hot_encode(data, features):
    for feature in features:
        if feature not in data.columns:
            raise ValueError(f"Feature '{feature}' not found in DataFrame.")

    # Perform one-hot encoding
    encoded_data = pd.get_dummies(data, columns=features, drop_first=False)

    return encoded_data


# def dimensionality_reduction(data):
#     # Standardize the data
#     scaler = StandardScaler()
#     data_scaled = scaler.fit_transform(data)
    
#     # Perform PCA
#     pca = PCA(n_components=n_components)
#     reduced_data = pca.fit_transform(data_scaled)
    
#     # Convert reduced data to DataFrame
#     reduced_df = pd.DataFrame(reduced_data, columns=[f"PC{i+1}" for i in range(n_components)])
#     return reduced_df





################### Data Cleaning ############################
def handle_missing_values(data, numerical_features):
    # Handle missing values
    # Option 1: Drop
    # data = data.dropna(axis=0)

    # Option 2: Use mean to impute values
    for col in numerical_features:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].mean())
    return data

    
    # # Standardize numerical features
    # scaler = StandardScaler()
    # numerical_columns = train_data.select_dtypes(include=[np.number]).columns
    # train_data[numerical_columns] = scaler.fit_transform(train_data[numerical_columns])
    # test_data[numerical_columns] = scaler.transform(test_data[numerical_columns])
    
    # # Encode categorical variables
    # encoder = OneHotEncoder(sparse=False, drop='first')
    # categorical_columns = train_data.select_dtypes(include=['object']).columns
    # train_encoded = encoder.fit_transform(train_data[categorical_columns])
    # test_encoded = encoder.transform(test_data[categorical_columns])


######################## EDA ##########################

def feature_histograms(data, numerical_columns, categorical_columns):
    
    # Plot histograms for numerical features
    for col in numerical_columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=data, x=col, kde=True,bins='auto', log_scale=True) #we used log-log scale to plot histogram as data was skewed
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()  # Display the plot

    # Plot bar plots for categorical features
    for col in categorical_columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=data, x=col)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()  # Display the plot

    print("Histograms and bar plots have been displayed.")
    
##Boxplots 
def plot_feature_boxplots(data, numerical_columns, categorical_columns):
    
    sns.set_style("whitegrid")
    
    # Loop through each numerical column and plot one boxplot at a time
    for col in numerical_columns:
        plt.figure(figsize=(7, 5))  # Create a new figure for each feature
        sns.boxplot(y=data[col])
        plt.title(f'Boxplot of {col}', fontsize=14)
        plt.xlabel('')
        plt.ylabel(col, fontsize=12)
        plt.show() 
    
##Heatmaps
        
def plot_correlation_heatmap(data, threshold=0.1):
    """
    Plots a heatmap showing only strong correlations above a certain threshold.

    Parameters:
    - data (pd.DataFrame): The dataset with numerical features.
    - threshold (float): Minimum absolute correlation value to include in the heatmap.
    """
    # Select numerical columns and compute correlation matrix
    numerical_data = data.select_dtypes(include=[np.number])
    correlation_matrix = numerical_data.corr()

    # Filter correlations based on the threshold
    strong_corr = correlation_matrix[
        (correlation_matrix.abs() >= threshold) & (correlation_matrix != 1.0)
    ]

    # Drop rows and columns where all values are NaN
    strong_corr = strong_corr.dropna(how='all', axis=0).dropna(how='all', axis=1)

    # Check if the resulting matrix is empty
    if strong_corr.empty:
        print(f"No correlations above the threshold of {threshold}.")
        return

    # Plot the filtered heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        strong_corr, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm", 
        cbar=True, 
        square=True,
        linewidths=0.5
    )
    plt.title(f"Filtered Correlation Heatmap (|correlation| >= {threshold})", fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.show()
