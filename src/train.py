import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from data_preprocessing import load_data, preprocess_data, get_features_and_target
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Model Mean Squared Error: {mse}')
    return model, mse, X_test


if __name__ == "__main__":
    # Set a unique experiment name
    experiment_name = "Mobile Price Prediction Experiment"
    mlflow.set_experiment(experiment_name)

    # Start MLFlow
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)

        # Load and preprocess data
        data = load_data('data/mobile-price-prediction-cleaned_data.csv')
        data = preprocess_data(data)
        X, y = get_features_and_target(data)
        # Train model and log metrics
        model, mse, X_test = train_model(X, y)
        # Log MSE
        mlflow.log_metric("mse", mse)

        # Prepare an input example
        input_example = X_test.head(1)
        # Infer model signature
        signature = infer_signature(X_test, model.predict(X_test))

        # Log model
        mlflow.sklearn.log_model(model, "model", signature=signature, input_example=input_example)

        # Save the trained modelpython src
        import joblib
        joblib.dump(model, 'model/mobilephone_price_linear_regression_model.pkl')
