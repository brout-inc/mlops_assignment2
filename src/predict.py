# prediction logic
# src/predict.py
import pandas as pd
import joblib


def load_model(model_filepath):
    model = joblib.load(model_filepath)
    return model


def make_predictions(model, data):
    predictions = model.predict(data)
    return predictions


if __name__ == "__main__":
    # Load model
    model = load_model('model/mobilephone_price_linear_regression_model.pkl')
    
    # Load new data
    new_data = pd.read_csv('data/new_mobile_data.csv')
    
    # Preprocess new data
    from data_preprocessing import preprocess_data
    new_data = preprocess_data(new_data)
    X_new, _ = get_features_and_target(new_data)
    
    # Make predictions
    predictions = make_predictions(model, X_new)
    print(predictions)
