import pytest
from fastapi.testclient import TestClient
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.app import app

client = TestClient(app)

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] in ["healthy", "unhealthy"]

def test_prediction():
    """Test prediction endpoint"""
    test_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    response = client.post("/predict", json=test_data)
    
    if response.status_code == 503:
        assert "Model not loaded" in response.json()["detail"]
    else:
        assert response.status_code == 200
        result = response.json()
        assert "prediction" in result
        assert "prediction_label" in result
        assert "confidence" in result
        assert result["prediction_label"] in ["setosa", "versicolor", "virginica"]

def test_invalid_input():
    """Test prediction with invalid input"""
    test_data = {
        "sepal_length": -1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 422

def test_metrics_endpoint():
    """Test metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "predictions_total" in response.text