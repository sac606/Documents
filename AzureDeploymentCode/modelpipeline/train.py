import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import optuna
from optuna.samplers import TPESampler

def objective(trial):
    """
    Optuna objective function for hyperparameter optimization
    """
    # Define hyperparameter space
    model_type = trial.suggest_categorical('model_type', ['random_forest', 'svm', 'logistic_regression'])
    
    if model_type == 'random_forest':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
        }
        model = RandomForestClassifier(**params, random_state=42)
        
    elif model_type == 'svm':
        params = {
            'C': trial.suggest_float('C', 0.1, 10.0, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
        }
        model = SVC(**params, random_state=42, probability=True)
        
    else:  # logistic_regression
        params = {
            'C': trial.suggest_float('C', 0.1, 10.0, log=True),
            'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'sag']),
            'max_iter': trial.suggest_int('max_iter', 100, 500),
        }
        model = LogisticRegression(**params, random_state=42)
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Use cross-validation for more robust evaluation
    score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    
    return score

def train_iris_model():
    """
    Train an Iris classification model with hyperparameter optimization using Optuna
    Returns:
        trained model with best parameters
    """
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Hyperparameter optimization with Optuna
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    )
    
    study.optimize(objective, n_trials=20, n_jobs=-1)  # Reduced trials for testing
    
    # Get best parameters
    best_params = study.best_params
    best_score = study.best_value
    
    print(f"Best hyperparameters: {best_params}")
    print(f"Best cross-validation accuracy: {best_score:.4f}")
    
    # Log Optuna study to MLflow
    mlflow.log_params(best_params)
    mlflow.log_metric("best_cv_accuracy", best_score)
    mlflow.log_param("n_trials", len(study.trials))
    
    # Train final model with best parameters
    if best_params['model_type'] == 'random_forest':
        final_model = RandomForestClassifier(
            n_estimators=best_params.get('n_estimators', 100),
            max_depth=best_params.get('max_depth', None),
            min_samples_split=best_params.get('min_samples_split', 2),
            min_samples_leaf=best_params.get('min_samples_leaf', 1),
            bootstrap=best_params.get('bootstrap', True),
            random_state=42
        )
    elif best_params['model_type'] == 'svm':
        final_model = SVC(
            C=best_params.get('C', 1.0),
            kernel=best_params.get('kernel', 'rbf'),
            gamma=best_params.get('gamma', 'scale'),
            random_state=42,
            probability=True
        )
    else:
        final_model = LogisticRegression(
            C=best_params.get('C', 1.0),
            solver=best_params.get('solver', 'lbfgs'),
            max_iter=best_params.get('max_iter', 100),
            random_state=42
        )
    
    final_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = final_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate metrics
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Log metrics to MLflow
    mlflow.log_metric("final_accuracy", final_accuracy)
    mlflow.log_metric("precision_avg", class_report['macro avg']['precision'])
    mlflow.log_metric("recall_avg", class_report['macro avg']['recall'])
    mlflow.log_metric("f1_avg", class_report['macro avg']['f1-score'])
    
    # Log model
    mlflow.sklearn.log_model(final_model, "iris_model")
    
    print(f"Final model trained with accuracy: {final_accuracy:.4f}")
    print(f"Model type: {best_params['model_type']}")
    
    return final_model

def train_baseline_model():
    """
    Train a baseline model without hyperparameter optimization
    """
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Log baseline metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("baseline_accuracy", accuracy)
    
    return model

if __name__ == "__main__":
    train_iris_model()