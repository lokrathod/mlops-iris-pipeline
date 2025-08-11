# Models Directory

This directory contains trained models after running the training pipeline.

## Files created after training:
- `scaler.pkl`: StandardScaler for feature normalization
- `best_model.pkl`: Best performing model selected by MLflow
- `data_hash.txt`: Hash of training data (for retraining detection)
- `last_training.txt`: Timestamp of last training

## To generate models:
```bash
python src/data_preprocessing.py
python src/train.py
```

Models are excluded from version control for size and security reasons.