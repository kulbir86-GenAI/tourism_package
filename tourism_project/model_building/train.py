
# for data manipulation
import pandas as pd

# for data preprocessing and pipeline creation
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# for model training and tuning
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# for model evaluation
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# for model serialization
import joblib

# for creating folders
import os

# for MLflow tracking
import mlflow


# ---------------------------------------------------------
# MLflow configuration
# ---------------------------------------------------------

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-tourism-training-experiment")


# ---------------------------------------------------------
# Load train/test data from previous preparation job
# ---------------------------------------------------------

Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")

ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

print("Training and testing datasets loaded successfully.")

print("Xtrain shape:", Xtrain.shape)
print("Xtest shape:", Xtest.shape)
print("ytrain shape:", ytrain.shape)
print("ytest shape:", ytest.shape)


# ---------------------------------------------------------
# Define numeric and categorical features
# ---------------------------------------------------------

numeric_features = [
    "Age",
    "CityTier",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome"
]

categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Passport",
    "Designation"
]


# ---------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    )
)


# ---------------------------------------------------------
# Define XGBoost Classifier
# ---------------------------------------------------------

xgb_model = xgb.XGBClassifier(
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"
)


# ---------------------------------------------------------
# Small grid so the pipeline runs fast on GitHub Actions
# ---------------------------------------------------------

param_grid = {
    "xgbclassifier__n_estimators": [50, 100],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.05, 0.1]
}


# ---------------------------------------------------------
# Pipeline
# ---------------------------------------------------------

model_pipeline = make_pipeline(
    preprocessor,
    xgb_model
)


# ---------------------------------------------------------
# MLflow run
# ---------------------------------------------------------

with mlflow.start_run():

    # -----------------------------------------------------
    # Grid Search
    # -----------------------------------------------------

    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=3,
        n_jobs=-1,
        scoring="f1"
    )

    print("Starting Grid Search...")

    grid_search.fit(
        Xtrain,
        ytrain
    )

    print("Grid Search completed.")


    # -----------------------------------------------------
    # Log all parameter combinations
    # -----------------------------------------------------

    results = grid_search.cv_results_

    for i in range(len(results["params"])):

        param_set = results["params"][i]

        mean_score = results["mean_test_score"][i]

        with mlflow.start_run(nested=True):

            mlflow.log_params(param_set)

            mlflow.log_metric(
                "mean_cv_f1",
                mean_score
            )


    # -----------------------------------------------------
    # Log best parameters
    # -----------------------------------------------------

    mlflow.log_params(
        grid_search.best_params_
    )


    # -----------------------------------------------------
    # Best model
    # -----------------------------------------------------

    best_model = grid_search.best_estimator_

    print(
        "Best params:",
        grid_search.best_params_
    )


    # -----------------------------------------------------
    # Predictions
    # -----------------------------------------------------

    y_pred_train = best_model.predict(Xtrain)

    y_pred_test = best_model.predict(Xtest)

    # Probability of ProdTaken = 1
    y_prob_train = best_model.predict_proba(Xtrain)[:, 1]

    y_prob_test = best_model.predict_proba(Xtest)[:, 1]


    # -----------------------------------------------------
    # Training metrics
    # -----------------------------------------------------

    train_accuracy = accuracy_score(
        ytrain,
        y_pred_train
    )

    train_precision = precision_score(
        ytrain,
        y_pred_train,
        zero_division=0
    )

    train_recall = recall_score(
        ytrain,
        y_pred_train,
        zero_division=0
    )

    train_f1 = f1_score(
        ytrain,
        y_pred_train,
        zero_division=0
    )

    train_roc_auc = roc_auc_score(
        ytrain,
        y_prob_train
    )


    # -----------------------------------------------------
    # Test metrics
    # -----------------------------------------------------

    test_accuracy = accuracy_score(
        ytest,
        y_pred_test
    )

    test_precision = precision_score(
        ytest,
        y_pred_test,
        zero_division=0
    )

    test_recall = recall_score(
        ytest,
        y_pred_test,
        zero_division=0
    )

    test_f1 = f1_score(
        ytest,
        y_pred_test,
        zero_division=0
    )

    test_roc_auc = roc_auc_score(
        ytest,
        y_prob_test
    )


    # -----------------------------------------------------
    # Log metrics to MLflow
    # -----------------------------------------------------

    mlflow.log_metrics({

        "train_accuracy": train_accuracy,
        "train_precision": train_precision,
        "train_recall": train_recall,
        "train_f1": train_f1,
        "train_roc_auc": train_roc_auc,

        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_roc_auc": test_roc_auc
    })


    # -----------------------------------------------------
    # Print metrics
    # -----------------------------------------------------

    print("\nTraining Metrics")
    print("-------------------------")
    print(f"Accuracy : {train_accuracy:.4f}")
    print(f"Precision: {train_precision:.4f}")
    print(f"Recall   : {train_recall:.4f}")
    print(f"F1 Score : {train_f1:.4f}")
    print(f"ROC-AUC  : {train_roc_auc:.4f}")

    print("\nTest Metrics")
    print("-------------------------")
    print(f"Accuracy : {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall   : {test_recall:.4f}")
    print(f"F1 Score : {test_f1:.4f}")
    print(f"ROC-AUC  : {test_roc_auc:.4f}")


    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------

    os.makedirs(
        "tourism_project/deployment",
        exist_ok=True
    )

    model_path = (
        "tourism_project/deployment/"
        "best_tourism_package_model_v1.joblib"
    )

    joblib.dump(
        best_model,
        model_path
    )


    # -----------------------------------------------------
    # Log model as MLflow artifact
    # -----------------------------------------------------

    mlflow.log_artifact(
        model_path,
        artifact_path="model"
    )

    print(
        f"\nModel saved to {model_path}"
    )

    print("Training completed successfully.")
