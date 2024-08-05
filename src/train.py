import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data


def preprocess_data(data):
    # Select features and target variable
    X = data[['Ratings', 'RAM', 'ROM', 'Mobile_Size', 'Primary_Cam', 'Selfi_Cam', 'Battery_Power']]
    y = data['Price']
    return X, y


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
    X, y = preprocess_data(data)
    model = train_model(X, y)
