# for data manipulation
import pandas as pd
# for data preprocessing and pipeline creation
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score, precision_score, f1_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
# for experiment tracking
import mlflow

# Set MLflow tracking URI (localhost for production)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("tourism-package-prediction-experiment")

# Initialize Hugging Face API
api = HfApi(token=os.getenv("HF_TOKEN"))

# Load processed datasets from Hugging Face Hub
# Update with your username
print("Loading processed datasets from Hugging Face Hub...")
Xtrain_path = "hf://datasets/Manish9119/tourism-package-prediction/Xtrain.csv"
Xtest_path = "hf://datasets/Manish9119/tourism-package-prediction/Xtest.csv"
ytrain_path = "hf://datasets/Manish9119/tourism-package-prediction/ytrain.csv"
ytest_path = "hf://datasets/Manish9119/tourism-package-prediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).values.ravel()
ytest = pd.read_csv(ytest_path).values.ravel()

print(f"Training set: {Xtrain.shape}, Test set: {Xtest.shape}")

# Define all numeric features (after encoding in prep.py)
numeric_features = list(Xtrain.columns)

# Calculate class weight to handle class imbalance
class_weight = sum(ytrain == 0) / sum(ytrain == 1)
print(f"Class weight (scale_pos_weight): {class_weight:.2f}")

# Define preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    remainder='passthrough'
)

# Define XGBoost model with class weight
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric='logloss'
)

# Define hyperparameter grid for tuning
param_grid = {
    'xgbclassifier__n_estimators': [100, 150, 200],
    'xgbclassifier__max_depth': [3, 5, 7],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__subsample': [0.6, 0.8, 1.0],
    'xgbclassifier__colsample_bytree': [0.6, 0.8, 1.0],
    'xgbclassifier__min_child_weight': [1, 3, 5],
}

# Create model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

print("Starting hyperparameter tuning with GridSearchCV...")

# Start MLflow run
with mlflow.start_run():
    # Perform hyperparameter tuning
    grid_search = GridSearchCV(
        model_pipeline, 
        param_grid, 
        cv=5, 
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(Xtrain, ytrain)
    
    print("Hyperparameter tuning completed.")
    
    # Log all parameter combinations and their scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]
        
        # Log each combination as a nested MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("cv_mean_f1_score", mean_score)
            mlflow.log_metric("cv_std_f1_score", std_score)
    
    # Log best parameters
    print(f"\nBest parameters: {grid_search.best_params_}")
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_f1_score", grid_search.best_score_)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Make predictions
    print("\nEvaluating model performance...")
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)
    
    # Calculate metrics for training set
    train_accuracy = accuracy_score(ytrain, y_pred_train)
    train_precision = precision_score(ytrain, y_pred_train)
    train_recall = recall_score(ytrain, y_pred_train)
    train_f1 = f1_score(ytrain, y_pred_train)
    
    # Calculate metrics for test set
    test_accuracy = accuracy_score(ytest, y_pred_test)
    test_precision = precision_score(ytest, y_pred_test)
    test_recall = recall_score(ytest, y_pred_test)
    test_f1 = f1_score(ytest, y_pred_test)
    
    # Log metrics to MLflow
    mlflow.log_metrics({
        "train_accuracy": train_accuracy,
        "train_precision": train_precision,
        "train_recall": train_recall,
        "train_f1_score": train_f1,
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1_score": test_f1
    })
    
    # Print evaluation results
    print("\n=== Model Performance ===")
    print(f"Training Set:")
    print(f"  Accuracy: {train_accuracy:.4f}")
    print(f"  Precision: {train_precision:.4f}")
    print(f"  Recall: {train_recall:.4f}")
    print(f"  F1-Score: {train_f1:.4f}")
    print(f"\nTest Set:")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  F1-Score: {test_f1:.4f}")
    
    # Print detailed classification reports
    print("\n=== Training Set Classification Report ===")
    print(classification_report(ytrain, y_pred_train))
    print("\n=== Test Set Classification Report ===")
    print(classification_report(ytest, y_pred_test))
    
    # Save the best model locally
    model_path = "best_tourism_model_v1.joblib"
    joblib.dump(best_model, model_path)
    print(f"\nModel saved locally: {model_path}")
    
    # Log the model as an artifact in MLflow
    mlflow.log_artifact(model_path, artifact_path="model")
    print("Model logged to MLflow as artifact")
    
    # Upload model to Hugging Face Hub
    print("\nUploading model to Hugging Face Hub...")
    repo_id = "Manish9119/tourism-package-model"
    repo_type = "model"
    
    # Check if model repository exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Model repository '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Model repository '{repo_id}' not found. Creating new repository...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Model repository '{repo_id}' created.")
    
    # Upload model file to Hugging Face
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print(f"Model uploaded to Hugging Face: {repo_id}")
    
print("\nModel training and registration completed successfully!")
