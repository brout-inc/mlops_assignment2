# Import Libraries
import mlflow
import mlflow.sklearn
import optuna
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#from sklearn.preprocessing import StandardScaler
#import pandas as pd
from data_preprocessing import load_data, preprocess_data


# Set up MLflow experiment
mlflow.set_experiment("wine_quality_experiment")  # Name of the experiment


# Define objective function for Optuna to tune hyperparameters
def objective(trial):
    model_type = trial.suggest_categorical('model_type', ['RandomForest', 'SVM'])
    # Hyperparameters for RandomForest
    if model_type == 'RandomForest':
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 5, 20)
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    # Hyperparameters for SVM
    else:
        C = trial.suggest_float('C', 0.1, 10.0)
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
        clf = SVC(C=C, gamma=gamma, random_state=42)
    # fit the model
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    # Log parameters and metrics in MLflow
    with mlflow.start_run():
        mlflow.log_param('model_type', model_type)
        mlflow.log_param('n_estimators', trial.params.get('n_estimators'))
        mlflow.log_param('max_depth', trial.params.get('max_depth'))
        mlflow.log_param('C', trial.params.get('C'))
        mlflow.log_param('gamma', trial.params.get('gamma'))
        # Log metrics
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('f1_score', f1)
        # Log the model
        mlflow.sklearn.log_model(clf, 'model')
    # Return accuracy
    return accuracy


# Main
if __name__ == "__main__":
    # wine quality dataset
    filepath="data/winequality-red.csv"
    # Load dataset
    df = load_data(filepath)
    # Preprocess dataset
    df = preprocess_data(df)
    # Split into features and Target
    X = df.drop('quality', axis=1)
    y = df['quality']
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Set up Optuna study for hyperparameter tuning
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    # Print the best trial
    print(f"Best trial: {study.best_trial.value}")
    print(f"Best hyperparameters: {study.best_trial.params}")
