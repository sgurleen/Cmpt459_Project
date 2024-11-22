import train
import utils
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.compose import ColumnTransformer


def main():
    # Read the training and test datasets
    train_path = "train/train.csv"
    test_path = "test/test.csv"

    train_data, test_data = utils.read_data(train_path, test_path)

    ############ Data Cleaning ##################

    # Data has 8 rows for each customer ID for different months. 
    # Dropping Nan values lead to 50 per loss in information
    # Since the data has 8 rows for each customer, taking its mode 
    # and filling the NaN or garbage values
    

    # Cleaning the data as there are many numeric value with '_', e.g. 23_
    train_data = train.filter_numeric_data(train_data)




    ############### Data Preprocessing ##########################


    # After evaluation Credit_Mix needs label encoding
    train_data = train.label_encoding(train_data, 'Credit_Mix', 'Payment_of_Min_Amount')
    # # Dropping NaN values in Credit Mix
    train_data = train_data.dropna(subset=['Credit_Mix'])

    utils.print_nan_counts(train_data)

    # Performing One hot encoding for 'Payment_behaviour'
    train_data = train.one_hot_encode(train_data, ['Payment_Behaviour'])

    # One-Hot encoding for Type_of_Loan changed the feature space to appx 6000
    # So going to split values here on ',' then create duplicate rows
    # Then apply one-hot 
    train_data = utils.split_and_duplicate_rows(train_data, 'Type_of_Loan')
    train_data = pd.get_dummies(train_data, columns=['Occupation'])
    # train_data = train.one_hot_encode(train_data, ['Occupation'])


    # ######################## Clustering ######################
    # Removing features that will not be useful for model 
    train_data = train_data.drop(columns=['ID', 'Customer_ID', 'Name', 'SSN', 'Month', 'Type_of_Loan'])

    train_data.drop_duplicates(
    keep='first',
    inplace=True
    )

    # Split between features and class labels
    y = train_data['Credit_Score']  # Extract the target variable
    X = train_data.drop(columns=['Credit_Score'])  # Drop the target variable to get the features
    
    robust_columns = ['Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']

    # Columns to apply StandardScaler (All columns except robust columns)
    standard_columns = [col for col in X.columns if col not in robust_columns]

    scaler = ColumnTransformer(
        transformers=[
            ('standard', StandardScaler(), standard_columns),
            ('robust', RobustScaler(), robust_columns)])

    # Apply to training data
    X = X.dropna()

    X.to_csv("cleaned.csv")
    X = scaler.fit_transform(X)

    # train.apply_dbscan(X)
    # train.apply_pca(X)
    labels_kmeans = train.apply_kmeans(X)

    train.check_cluster_performance(X, labels_kmeans)
    
    

    
main()
