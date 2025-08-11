from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import json
import os
import logging
from pythonjsonlogger import jsonlogger
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import PlainTextResponse
import time
import subprocess

from api.schemas import IrisFeatures, PredictionResponse, HealthResponse

# Configure JSON logging
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

# Prometheus metrics
prediction_counter = Counter("predictions_total", "Total number of predictions")
prediction_histogram = Histogram("prediction_duration_seconds", "Prediction duration")
prediction_class_counter = Counter(
    "predictions_by_class", "Predictions by class", ["class_name"]
)

# Initialize FastAPI
app = FastAPI(title="Iris Classification API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler
try:
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    MODEL_LOADED = True
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    MODEL_LOADED = False
    model = None
    scaler = None

# Class mapping
CLASS_NAMES = {0: "setosa", 1: "versicolor", 2: "virginica"}


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if MODEL_LOADED else "unhealthy",
        model_loaded=model is not None,
        scaler_loaded=scaler is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: IrisFeatures):
    """Make prediction on iris features"""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        # Create DataFrame with proper feature names
        feature_names = [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ]
        feature_df = pd.DataFrame(
            [
                [
                    features.sepal_length,
                    features.sepal_width,
                    features.petal_length,
                    features.petal_width,
                ]
            ],
            columns=feature_names,
        )

        # Scale features
        feature_scaled = scaler.transform(feature_df)

        # Make prediction
        prediction = model.predict(feature_scaled)[0]
        prediction_proba = model.predict_proba(feature_scaled)[0]
        confidence = float(np.max(prediction_proba))

        # Update metrics
        prediction_counter.inc()
        prediction_class_counter.labels(class_name=CLASS_NAMES[prediction]).inc()

        # Log prediction
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "features": features.dict(),
            "prediction": int(prediction),
            "prediction_label": CLASS_NAMES[prediction],
            "confidence": confidence,
            "duration": time.time() - start_time,
        }
        logger.info("prediction_made", extra=log_entry)

        # Save to file (simple logging)
        os.makedirs("logs", exist_ok=True)
        with open("logs/predictions.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Record duration
        prediction_histogram.observe(time.time() - start_time)

        return PredictionResponse(
            prediction=int(prediction),
            prediction_label=CLASS_NAMES[prediction],
            confidence=confidence,
            features=features.dict(),
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()


@app.post("/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    """Trigger model retraining in background"""

    def retrain_model():
        try:
            # Run retraining script
            result = subprocess.run(
                ["python", "src/retrain.py"], capture_output=True, text=True
            )

            if result.returncode == 0:
                logger.info("Model retraining completed successfully")
            else:
                logger.error(f"Model retraining failed: {result.stderr}")

        except Exception as e:
            logger.error(f"Retraining error: {e}")

    # Add task to background
    background_tasks.add_task(retrain_model)

    return {"message": "Retraining triggered in background"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
