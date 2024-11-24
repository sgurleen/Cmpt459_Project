import train
import utils
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.compose import ColumnTransformer
import eval
from sklearn.neighbors import KNeighborsClassifier

def main():
    # Read the training and test datasets
    train_path = "./train/train.csv"
    test_path = "./test/test.csv"

    train_data, test_data = utils.read_data(train_path, test_path)


    ############ Data Cleaning ##################

    # Data has 8 rows for each customer ID for different months. 
    # Dropping Nan values lead to 50 per loss in information
    # Since the data has 8 rows for each customer, taking its mode 
    # and filling the NaN or garbage values

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

    credit_mapping = {"Poor": 0, "Standard": 1, "Good": 2}
    y = y.map(credit_mapping)
    # print(y.value_counts())
    robust_columns = ['Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']

    # Columns to apply StandardScaler (All columns except robust columns)
    standard_columns = [col for col in X.columns if col not in robust_columns]

    scaler = ColumnTransformer(
        transformers=[
            ('standard', StandardScaler(), standard_columns),
            ('robust', RobustScaler(), robust_columns)])

    # Apply to training data
    X = X.dropna()
    print(X.shape)
    X = scaler.fit_transform(X)

    X_filtered = utils.pick_first_row_every_8(X)

    # There are 3 class Good, Poor, and Standard
    n_clusters = 3

    cluster_labels, kmeans_model = train.apply_kmeans(X_filtered, n_clusters)
    # cluster_labels = train.perform_dbscan(X_filtered, eps=8, min_samples=20)
    
    # Perform PCA for visualization
    pca_data, pca_model = train.perform_pca(X_filtered, n_components=3)

    # Plot PCA clusters
    train.plot_pca_clusters_2D(pca_data, cluster_labels, n_clusters)
    
    # Evaluate silhouette score
    silhouette = train.evaluate_silhouette_score(X_filtered, cluster_labels)
    print(f"Silhouette Score: {silhouette:.2f}")
 
    
    ################### KNN Neighbors ##################################
    
    # X_train, X_test, y_train, y_test = utils.split_dataset(X, y)

    # knn = KNeighborsClassifier(n_neighbors=20)

    # ################### Cross-validation ##############################
    # cv_scores = eval.perform_cross_validation(knn, X_train, y_train, cv=5)
    # print("Cross-Validation Scores:", cv_scores)

    # knn.fit(X_train, y_train)
    # y_pred = knn.predict(X_test)
    # y_proba = knn.predict_proba(X_test)

    # metrics = eval.evaluate_model(y_test, y_pred, y_proba)
    # print("Evaluation Metrics:", metrics)

    # eval.plot_confusion_matrix(y_test, y_pred)
    # if y_proba is not None:
    #     eval.plot_multiclass_roc_curve(y_test, y_proba, ["Poor", "Standard", "Good"])

    # ###################### Hyperparameter tuning ######################
    # param_grid = {"n_neighbors": [10, 20, 50, 70, 100], "weights": ["uniform", "distance"]}
    # grid_search = eval.perform_grid_search(knn, param_grid, X_train, y_train)
    # print("Best Parameters:", grid_search.best_params_)

    
    ################################ EDA ################################
    # Make histograms
    numerical_columns = train_data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = train_data.select_dtypes(include=['object', 'category']).columns.tolist()

    
    # Plot histograms and show them
    # train.feature_histograms(train_data, numerical_columns, categorical_columns)
    
    # Plot boxplors
    # log_transformed_data = utils.apply_transformation(train_data, numerical_columns)
    # train.plot_feature_boxplots(train_data, numerical_columns, categorical_columns)
    
    # heatmaps
    # train.plot_correlation_heatmap(train_data)

    

main()
