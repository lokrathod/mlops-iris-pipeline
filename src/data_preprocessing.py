import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_preprocess_data():
    """Load Iris dataset and perform preprocessing"""
    logger.info("Loading Iris dataset...")

    # Load data
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")

    # Save raw data
    os.makedirs("data/raw", exist_ok=True)
    raw_data = pd.concat([X, y], axis=1)
    raw_data.to_csv("data/raw/iris.csv", index=False)
    logger.info(f"Raw data saved: {raw_data.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save processed data
    os.makedirs("data/processed", exist_ok=True)
    train_data = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_data["target"] = y_train.values
    train_data.to_csv("data/processed/train.csv", index=False)

    test_data = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_data["target"] = y_test.values
    test_data.to_csv("data/processed/test.csv", index=False)

    # Save scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")

    logger.info("Data preprocessing completed!")
    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    load_and_preprocess_data()
