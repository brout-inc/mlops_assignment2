import pytest
from src.train import load_data, preprocess_data

def test_load_data():
    data = load_data('data/winequality-red.csv')
    assert not data.empty

def test_preprocess_data():
    data = load_data('data/winequality-red.csv')
    data = preprocess_data(data)
    assert not data.empty
