import pytest
from src.train import load_data, preprocess_data, train_model

def test_load_data():
    data = load_data('data/mobile-price-prediction-cleaned_data.csv')
    assert not data.empty

def test_preprocess_data():
    data = load_data('data/mobile-price-prediction-cleaned_data.csv')
    X, y = preprocess_data(data)
    assert X.shape[0] == y.shape[0]

def test_train_model():
    data = load_data('data/mobile-price-prediction-cleaned_data.csv')
    X, y = preprocess_data(data)
    model = train_model(X, y)
    assert model is not None
