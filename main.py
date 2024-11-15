import train
import utils
import numpy as np
import pandas as pd


def main():
    # Read the training and test datasets
    train_path = "train/train.csv"
    test_path = "test/test.csv"

    train_data, test_data = utils.read_data(train_path, test_path)

    # Removing all NaN values to be able to differntiate in Categorical
    # and Numerical Variables
    no_nan_train_data = train_data.dropna()  

    # Cleaning the data as there are many numeric value with '_', e.g. 23_
    clean_train_data = train.filter_numeric_data(no_nan_train_data)

    # Get Numerical and Categorical Features
    numerical_columns = clean_train_data.select_dtypes(include=[np.number]).columns
    print("Numerical Features:")
    print(numerical_columns.tolist())
    categorical_columns = clean_train_data.select_dtypes(include=['object', 'category']).columns
    print("\nCategorical Features:")
    print(categorical_columns.tolist())

    # Dropping Nan leads to 50 per data loss
    # Using mean to impute values instead
    train_data = train.filter_numeric_data(train_data)
    train_data = train.handle_missing_values(train_data, numerical_columns)
    train_data = train.encode_categorical_to_numerical(train_data, 'Credit_History_Age')

    # For now imputing these values, might consider to drop them
    train_data = train.handle_missing_values(train_data, ['Credit_History_Age'])

    # After evaluation Credit_Mix needs label encoding
    train_data = train.label_encoding(train_data, 'Credit_Mix', 'Payment_of_Min_Amount')
    # Dropping NaN values in Credit Mix
    train_data = train_data.dropna(subset=['Credit_Mix'])
    
    

    print(train_data.shape)

main()
