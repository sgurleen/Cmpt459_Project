from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV

def perform_cross_validation(model, X, y, cv=5):

    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores


def evaluate_model(y_test, y_pred, y_proba=None):
    multi_class='ovr'
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1-Score": f1_score(y_test, y_pred, average='weighted'),
    }
    if len(y_proba.shape) == 1:
        y_proba = y_proba.reshape(-1, 1)

    if y_proba is not None:
        metrics["AUC-ROC"] = roc_auc_score(y_test, y_proba, multi_class=multi_class, average='weighted')
    return metrics

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig('Knn/confusion_matrix.png')

def plot_multiclass_roc_curve(y_test, y_proba, class_names):
    # Binarize the output
    y_test_binarized = label_binarize(y_test, classes=range(len(class_names)))
    n_classes = y_test_binarized.shape[1]

    plt.figure(figsize=(10, 7))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {class_names[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=2)  # Diagonal line
    plt.title("Multiclass ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig('Knn/roc_curve.png')

def perform_grid_search(model, param_grid, X_train, y_train, cv=5):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search
