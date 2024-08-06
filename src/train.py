import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from data_preprocessing import load_data, preprocess_data, get_features_and_target


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Model Mean Squared Error: {mse}')
    return model


if __name__ == "__main__":
    data = load_data('data/mobile-price-prediction-cleaned_data.csv')
    data = preprocess_data(data)
    X, y = get_features_and_target(data)
    model = train_model(X, y)

    # Save the trained model
    import joblib
    joblib.dump(model, 'model/mobilephone_price_linear_regression_model.pkl')
