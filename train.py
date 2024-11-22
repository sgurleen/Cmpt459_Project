import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import utils
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

########## Data Preprocessing ##################
def filter_numeric_data(data):

    # Age has values with _ char e.g 23_
    data = utils.keep_only_numeric_values(data, 'Age')
    data['Age'] = data['Age'].where((data['Age'] >= 14) & (data['Age'] <= 56), np.nan)
    data = utils.forward_backward_filling(data, 'Age', 'Customer_ID')

    
    data['Occupation'] = data['Occupation'].replace('_______', pd.NA)
    data = utils.fill_with_group_mode(data, 'Occupation', 'Customer_ID')

    data = utils.keep_only_numeric_values(data, 'Annual_Income')
    data['Annual_Income'] = data['Annual_Income'].astype('float')
    data = utils.fill_with_group_mode(data, 'Annual_Income', 'Customer_ID')
    
    data.loc[data['Num_Bank_Accounts'].isin([-1, 0]), 'Num_Bank_Accounts'] = np.nan
    data.loc[data['Num_Bank_Accounts'] > 11, 'Num_Bank_Accounts'] = np.nan
    data = utils.fill_with_group_mode(data, 'Num_Bank_Accounts', 'Customer_ID')
    
    data['Num_Bank_Accounts'].fillna(data['Num_Bank_Accounts'].mean(), inplace=True)

    data = utils.fill_with_group_mode(data, 'Monthly_Inhand_Salary', 'Customer_ID') 
    
    data = utils.fill_with_group_mode(data, 'Num_Credit_Card', 'Customer_ID')

    data = utils.fill_with_group_mode(data, 'Interest_Rate', 'Customer_ID')

    data = utils.keep_only_numeric_values(data, 'Num_of_Loan')
    
    data = utils.fill_with_group_mode(data, 'Num_of_Loan', 'Customer_ID')

    data['Type_of_Loan'].fillna('Zero Loan', inplace=True)

    data.loc[data.Delay_from_due_date < 0, 'Delay_from_due_date'] = np.nan
    data = utils.fill_with_group_mode(data, 'Delay_from_due_date', 'Customer_ID')

    data = utils.keep_only_numeric_values(data, 'Num_of_Delayed_Payment')
    data = utils.forward_backward_filling(data, 'Num_of_Delayed_Payment', 'Customer_ID')

    data['Changed_Credit_Limit'] = data['Changed_Credit_Limit'].replace('_', pd.NA)
    # data['Changed_Credit_Limit'] = data['Changed_Credit_Limit'].round(2)
    data = utils.keep_only_numeric_values(data, 'Changed_Credit_Limit')
    data = utils.fill_with_group_mode(data, 'Changed_Credit_Limit', 'Customer_ID')

    data = utils.forward_backward_filling(data, 'Num_Credit_Inquiries', 'Customer_ID')

    data = utils.fill_with_group_mode(data, 'Credit_Mix', 'Customer_ID')

    data = utils.keep_only_numeric_values(data, 'Outstanding_Debt')

    data = utils.convert_to_months(data, 'Credit_History_Age')
    data = utils.forward_backward_filling(data, 'Credit_History_Age', 'Customer_ID')
    
    # data = data.drop(columns='Credit_History_Age')

    data['Amount_invested_monthly'] = data.Amount_invested_monthly.replace('__10000__', np.nan)
    data = utils.forward_backward_filling(data, 'Amount_invested_monthly', 'Customer_ID')
    

    data['Payment_Behaviour'] = data.Payment_Behaviour.replace('!@9#%8', 'Unknown')

    data.loc[data.Monthly_Balance == '__-333333333333333333333333333__', 'Monthly_Balance'] = np.nan

    # data['Monthly_Balance'] = pd.to_numeric(data['Monthly_Balance'], errors='coerce')
    data['Monthly_Balance'] = data['Monthly_Balance'].astype(float)
    data = utils.forward_backward_filling(data, 'Monthly_Balance', 'Customer_ID')


    return data


def encode_categorical_to_numerical(data, feature):
    encoded_data = utils.convert_to_months(data, feature)

    return encoded_data

def label_encoding(data, feature1, feature2):

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

def scale_features(X):
    """
    Scale the features of the given dataset using StandardScaler.

    Parameters:
    - X: pd.DataFrame or np.ndarray - The data to be scaled (numeric only).

    Returns:
    - pd.DataFrame - The scaled version of the dataset.
    """
    # Instantiate the scaler
    scaler = StandardScaler()

    # Fit the scaler to the data and transform it
    scaled_data = scaler.fit_transform(X)

    # Convert to DataFrame with original column names
    scaled_df = pd.DataFrame(scaled_data, columns=X.columns)

    return scaled_df


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


####################### Clustering ###########################
def apply_dbscan(data, eps=0.5, min_samples=5):
    """
    Apply DBSCAN clustering on the given dataset and plot the resulting clusters.

    Parameters:
    - data: pd.DataFrame or np.ndarray - The data to cluster (numeric only).
    - eps: float - The maximum distance between two samples for them to be considered as in the same neighborhood.
    - min_samples: int - The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
    - pd.Series - Cluster labels for each data point.
    """
    # Instantiate DBSCAN with specified parameters
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    # Fit DBSCAN to the data and predict cluster labels
    labels = dbscan.fit_predict(data)

    # Return the labels as a Pandas Series for better interpretability
    labels_series = pd.Series(labels, name='Cluster_Label')

    # Plotting the clusters
    plt.figure(figsize=(10, 6))
    unique_labels = np.unique(labels)
    palette = sns.color_palette("hsv", len(unique_labels))

    for label, color in zip(unique_labels, palette):
        if label == -1:
            # Noise points are labeled as -1
            color = 'k'
            marker = 'x'
        else:
            marker = 'o'
        
        cluster_points = data[labels == label]
        plt.scatter(cluster_points.iloc[:, 0], cluster_points.iloc[:, 1], c=[color], label=f'Cluster {label}' if label != -1 else 'Noise', marker=marker)

    plt.title('DBSCAN Clustering Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

    return labels_series

def apply_kmeans(X, n_clusters=5, random_state=42):
    """
    Apply K-Means clustering to the given dataset.

    Parameters:
    - X: pd.DataFrame or np.ndarray - The data to cluster (numeric only).
    - n_clusters: int - The number of clusters to form. Default is 3.
    - random_state: int - Random seed for reproducibility.

    Returns:
    - pd.Series - Cluster labels for each data point.
    """
    # Instantiate the KMeans model with specified parameters
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)

    # Fit the KMeans model to the data and predict the cluster labels
    labels = kmeans.fit_predict(X)

    # Return the labels as a Pandas Series for better interpretability
    return pd.Series(labels, name='Cluster_Label')

def apply_pca(data, n_components=2):
    """
    Apply PCA (Principal Component Analysis) to reduce the dimensionality of the dataset and plot the explained variance.

    Parameters:
    - data: pd.DataFrame or np.ndarray - The data to reduce (numeric only).
    - n_components: int - The number of principal components to keep.

    Returns:
    - pd.DataFrame - Transformed data with the specified number of principal components.
    """
    # Instantiate PCA with the specified number of components
    pca = PCA(n_components=n_components)

    # Fit and transform the data using PCA
    pca_transformed = pca.fit_transform(data)

    # Create a DataFrame with the transformed data
    pca_columns = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(pca_transformed, columns=pca_columns)

    # Plotting the explained variance by each principal component
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, n_components + 1), pca.explained_variance_ratio_ * 100, alpha=0.7, align='center')
    plt.xlabel('Principal Component')
    plt.ylabel('Percentage of Explained Variance')
    plt.title('Explained Variance by Principal Components')
    plt.xticks(range(1, n_components + 1))
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()

    return pca_df


def check_cluster_performance(data, labels):
    """
    Evaluate the clustering performance using silhouette score.

    Parameters:
    - data: pd.DataFrame or np.ndarray - The data used for clustering (numeric only).
    - labels: pd.Series or np.ndarray - The cluster labels assigned by the clustering algorithm.

    Returns:
    - float - The silhouette score of the clustering, ranging from -1 to 1.
    """
    # Calculate the silhouette score
    score = silhouette_score(data, labels)
    
    # Print the silhouette score
    print(f'Silhouette Score: {score:.2f}')
    
    return score





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
