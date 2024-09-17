import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_preprocessing import load_data, preprocess_data, get_features_and_target
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature


if __name__ == "__main__":
    # Load and preprocess data
    data = load_data('data/mobile-price-prediction-cleaned_data.csv')
    data = preprocess_data(data)
    X, y = get_features_and_target(data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model and parameters for GridSearchCV
    model = LinearRegression()
    param_grid = {
        'fit_intercept': [True, False],
        'copy_X': [True, False]
    }

    # Perform hyperparameter tuning
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print("Best parameters found: ", best_params)

    # Evaluate the best model
    predictions = best_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("MSE: ", mse)
    print("MAE: ", mae)
    print("R2 Score: ", r2)

    # Log metrics and model
    experiment_name = "Mobile Price Prediction Experiment"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "mse": mse,
            "mae": mae,
            "r2_score": r2
        })
        # Prepare an input example
        input_example = X_test.head(1)
        # Infer model signature
        signature = infer_signature(X_test, best_model.predict(X_test))

        # Log model
        mlflow.sklearn.log_model(best_model, "best_model", signature=signature, input_example=input_example)
        #mlflow.sklearn.log_model(best_model, "best_model")
