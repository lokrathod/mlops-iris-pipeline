from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import joblib
import logging
from data_preprocessing import load_and_preprocess_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
    }

    return metrics


def train_models():
    """Train multiple models and track with MLflow"""
    # Set MLflow tracking URI - always use SQLite
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # Create or get experiment
    try:
        mlflow.create_experiment("iris-classification")
    except Exception:
        # Experiment already exists
        pass

    mlflow.set_experiment("iris-classification")

    # Load data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Model 1: Logistic Regression
    with mlflow.start_run(run_name="logistic-regression"):
        logger.info("Training Logistic Regression...")

        # Parameters
        params = {"max_iter": 1000, "random_state": 42, "solver": "lbfgs"}

        # Train model
        lr_model = LogisticRegression(**params)
        lr_model.fit(X_train, y_train)

        # Evaluate
        lr_metrics = evaluate_model(lr_model, X_test, y_test)

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metrics(lr_metrics)
        mlflow.sklearn.log_model(lr_model, "model")

        logger.info(f"Logistic Regression metrics: {lr_metrics}")

    # Model 2: Random Forest
    with mlflow.start_run(run_name="random-forest"):
        logger.info("Training Random Forest...")

        # Parameters
        params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}

        # Train model
        rf_model = RandomForestClassifier(**params)
        rf_model.fit(X_train, y_train)

        # Evaluate
        rf_metrics = evaluate_model(rf_model, X_test, y_test)

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metrics(rf_metrics)
        mlflow.sklearn.log_model(rf_model, "model")

        logger.info(f"Random Forest metrics: {rf_metrics}")

    # Register best model
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("iris-classification")
    runs = client.search_runs(experiment.experiment_id)

    best_run = max(runs, key=lambda r: r.data.metrics["accuracy"])
    model_uri = f"runs:/{best_run.info.run_id}/model"

    mlflow.register_model(model_uri, "iris-classifier")
    logger.info(
        f"Best model registered with accuracy: {best_run.data.metrics['accuracy']}"
    )

    # Save best model locally
    best_model = mlflow.sklearn.load_model(model_uri)
    joblib.dump(best_model, "models/best_model.pkl")

    logger.info("Model training completed successfully!")


if __name__ == "__main__":
    train_models()
