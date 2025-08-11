import pytest
import numpy as np
import pandas as pd
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import load_and_preprocess_data

def test_data_preprocessing():
    """Test data preprocessing function"""
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Check shapes
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[1] == 4  # 4 features
    
    # Check scaling (mean ~0, std ~1)
    assert np.abs(np.mean(X_train, axis=0)).max() < 0.1
    assert np.abs(np.std(X_train, axis=0) - 1).max() < 0.1

def test_data_files_created():
    """Test that data files are created"""
    load_and_preprocess_data()
    
    assert os.path.exists('data/raw/iris.csv')
    assert os.path.exists('data/processed/train.csv')
    assert os.path.exists('data/processed/test.csv')
    assert os.path.exists('models/scaler.pkl')