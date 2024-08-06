# data processing steps, if any
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(filepath):
    data = pd.read_csv(filepath)
    return data


def preprocess_data(data):
    # Example preprocessing: scaling numerical features
    scaler = StandardScaler()
    data[['RAM', 'ROM', 'Mobile_Size', 'Primary_Cam', 'Selfi_Cam', 'Battery_Power']] = scaler.fit_transform(
        data[['RAM', 'ROM', 'Mobile_Size', 'Primary_Cam', 'Selfi_Cam', 'Battery_Power']]
    )
    return data


def get_features_and_target(data):
    X = data[['Ratings', 'RAM', 'ROM', 'Mobile_Size', 'Primary_Cam', 'Selfi_Cam', 'Battery_Power']]
    y = data['Price']
    return X, y
