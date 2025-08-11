import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import joblib
import os

def test_model_exists():
    """Test if model file exists"""
    assert os.path.exists('models/best_model.pkl'), "Model file not found"
    assert os.path.exists('models/scaler.pkl'), "Scaler file not found"

def test_model_prediction():
    """Test model makes valid predictions"""
    # Load model and scaler
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    # Create test data
    feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    test_data = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=feature_names)
    
    # Scale and predict
    scaled_data = scaler.transform(test_data)
    prediction = model.predict(scaled_data)
    
    # Check prediction is valid
    assert prediction[0] in [0, 1, 2], "Invalid prediction class"
    
def test_model_probability():
    """Test model returns valid probabilities"""
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    test_data = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=feature_names)
    
    scaled_data = scaler.transform(test_data)
    probabilities = model.predict_proba(scaled_data)
    
    # Check probabilities sum to 1
    assert np.isclose(probabilities.sum(), 1.0), "Probabilities don't sum to 1"
    assert all(p >= 0 and p <= 1 for p in probabilities[0]), "Invalid probability values"