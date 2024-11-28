import preprocessing
import utils
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.compose import ColumnTransformer
import eval
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

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

    train_data = preprocessing.filter_numeric_data(train_data)


    ############### Data Preprocessing ##########################

    # After evaluation Credit_Mix needs label encoding
    train_data = preprocessing.label_encoding(train_data, 'Credit_Mix', 'Payment_of_Min_Amount')
    # # Dropping NaN values in Credit Mix
    train_data = train_data.dropna(subset=['Credit_Mix'])

    utils.print_nan_counts(train_data)

    # Performing One hot encoding for 'Payment_behaviour'
    train_data = preprocessing.one_hot_encode(train_data, ['Payment_Behaviour'])

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

    ## Detect and remove outliers
    # train_data = utils.detect_outliers_per_column_and_save_plots(X, method="elliptic_envelope", contamination=0.05, output_dir=None)

    ## Mapping credit category to numbers
    credit_mapping = {"Poor": 0, "Standard": 1, "Good": 2}
    y = y.map(credit_mapping)
    robust_columns = ['Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance', 'Annual_Income']

    # Columns to apply StandardScaler (All columns except robust columns)
    standard_columns = [col for col in X.columns if col not in robust_columns]

    # Constructing transformer for robust and standard scaler
    scaler = ColumnTransformer(
        transformers=[
            ('standard', StandardScaler(), standard_columns),
            ('robust', RobustScaler(), robust_columns)])

    # Apply to training data
    X = X.dropna()
    print(X.shape)
    X = scaler.fit_transform(X)

    # Filtering every 1st row out
    X_filtered = utils.pick_first_row_every_8(X)

    # There are 3 class Good, Poor, and Standard
    n_clusters = 3

    cluster_labels, kmeans_model = preprocessing.apply_kmeans(X_filtered, n_clusters)
    # cluster_labels = train.perform_dbscan(X_filtered, eps=8, min_samples=50)
    # cluster_labels = train.perform_hierarchical_clustering(X_filtered)
    
    # Perform PCA for visualization
    pca_data, pca_model = preprocessing.perform_pca(X_filtered, n_components=3)

    # Plot PCA clusters
    preprocessing.plot_pca_clusters_2D(pca_data, cluster_labels, n_clusters)
    
    # Evaluate silhouette score
    silhouette = preprocessing.evaluate_silhouette_score(X_filtered, cluster_labels)
    print(f"Silhouette Score: {silhouette:.2f}")
 
    
    ################### KNN Neighbors ##################################
    
    # Train-test split
    X_train, X_test, y_train, y_test = utils.split_dataset(X, y)

    # SMOTE based class balancing
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    X_test, y_test = smote.fit_resample(X_test, y_test)


    # Defining clssification modela
    model = KNeighborsClassifier(n_neighbors=20)
    # model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=None, min_samples_split=2, min_samples_leaf=1)
    # model = AdaBoostClassifier(n_estimators=100, algorithm="SAMME",)

    # Feature Selection using RFE
    # rfe = RFE(estimator=model, n_features_to_select=20)  # Selecting top 10 features
    # X_train = rfe.fit_transform(X_train, y_train)
    # X_test = rfe.transform(X_test)

    # ################### Cross-validation ##############################
    cv_scores = eval.perform_cross_validation(model, X_train, y_train, cv=5)
    print("Cross-Validation Scores:", cv_scores)

    # Training classification model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics = eval.evaluate_model(y_test, y_pred, y_proba)
    print("Evaluation Metrics:", metrics)

    eval.plot_confusion_matrix(y_test, y_pred)
    if y_proba is not None:
        eval.plot_multiclass_roc_curve(y_test, y_proba, ["Poor", "Standard", "Good"])

    # ###################### Hyperparameter tuning ######################
    # param_grid = {"n_neighbors": [10, 20, 50, 70, 100], "weights": ["uniform", "distance"]}
    # grid_search = eval.perform_grid_search(knn, param_grid, X_train, y_train)
    # print("Best Parameters:", grid_search.best_params_)

    # param_grid = {
    #     "n_estimators": [50, 100, 200],
    #     "max_depth": [None, 10, 20, 30],
    #     "min_samples_split": [2, 5, 10],
    #     "min_samples_leaf": [1, 2, 4]
    # }

    # grid_search = eval.perform_grid_search(model, param_grid, X_train, y_train)
    # grid_search.fit(X_train, y_train)
    # print("Best Parameters:", grid_search.best_params_)
    
    ################################ EDA ################################
    # # Make histograms
    # numerical_columns = train_data.select_dtypes(include=[np.number]).columns.tolist()

    # # Plot histograms and show them
    # train.feature_histograms(train_data, numerical_columns)
    
    # # Plot boxplots
    # # log_transformed_data = utils.apply_transformation(train_data, numerical_columns)
    # train.scatter_plot_individual_features(train_data, numerical_columns)
    # train.plot_feature_boxplots(train_data, numerical_columns)
    
    # # heatmaps
    # preprocessing.plot_correlation_heatmap(train_data)


    

main()
