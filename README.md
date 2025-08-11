# MLOps Pipeline for Iris Classification 🌺

[![CI/CD Pipeline](https://github.com/lokrathod/mlops-iris-pipeline/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/lokrathod/mlops-iris-pipeline/actions)
[![Docker Hub](https://img.shields.io/docker/pulls/rathodlok/iris-classifier)](https://hub.docker.com/r/rathodlok/iris-classifier)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

## 🎯 Overview

A production-ready MLOps pipeline implementing end-to-end machine learning workflow for Iris flower classification. This project demonstrates industry best practices including automated training, containerized deployment, real-time monitoring, and continuous integration/deployment.

## ✨ Features

- **Automated ML Pipeline** - End-to-end automation from data preprocessing to model deployment
- **Experiment Tracking** - MLflow integration for reproducible machine learning
- **Containerized Architecture** - Docker-based microservices for scalability
- **Real-time Monitoring** - Prometheus metrics with Grafana visualization
- **CI/CD Integration** - GitHub Actions for automated testing and deployment
- **Auto-retraining** - Intelligent triggers based on data drift and performance
- **RESTful API** - FastAPI with automatic documentation and validation
- **Production Ready** - Health checks, error handling, and logging

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Git

### Using Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/mlops-iris-pipeline.git
cd mlops-iris-pipeline

# Start all services
docker-compose up -d

# Check service health
docker ps

# Access services
# API: http://localhost:8000
# API Docs: http://localhost:8000/docs
# MLflow: http://localhost:5000
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
```

### Local Development

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run data preprocessing
python src/data_preprocessing.py

# Train models
python src/train.py

# Start API server
uvicorn api.app:app --reload
```

## 📁 Project Structure

```
mlops-iris-pipeline/
├── api/                    # FastAPI application
│   ├── app.py             # Main API with endpoints
│   └── schemas.py         # Pydantic models for validation
├── src/                    # ML pipeline source code
│   ├── data_preprocessing.py
│   ├── train.py           # Model training with MLflow
│   ├── retrain.py         # Manual retraining logic
│   └── auto_retrain_monitor.py
├── tests/                  # Unit and integration tests
├── scripts/                # Utility and deployment scripts
├── grafana/                # Grafana dashboards and config
├── models/                 # Trained model artifacts
├── data/                   # Data directory
├── logs/                   # Application logs
├── docker-compose.yml      # Multi-container orchestration
├── Dockerfile              # Container definition
├── prometheus.yml          # Prometheus configuration
└── requirements.txt        # Python dependencies
```

## 🔧 Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    Data     │────▶│   Training  │────▶│   MLflow    │
│ Processing  │     │  Pipeline   │     │  Registry   │
└─────────────┘     └─────────────┘     └─────────────┘
                            │
                            ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Docker    │◀────│   FastAPI   │────▶│ Prometheus  │
│ Containers  │     │   Service   │     │  Metrics    │
└─────────────┘     └─────────────┘     └─────────────┘
                            │                    │
                            ▼                    ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   GitHub    │     │    Users    │     │   Grafana   │
│   Actions   │     │    (API)    │     │ Dashboard   │
└─────────────┘     └─────────────┘     └─────────────┘
```

## 📊 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/docs` | Interactive API documentation |
| POST | `/predict` | Make iris classification prediction |
| GET | `/metrics` | Prometheus metrics endpoint |
| POST | `/retrain` | Trigger model retraining |

### Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

### Example Response

```json
{
  "prediction": 0,
  "prediction_label": "setosa",
  "confidence": 0.98,
  "features": {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }
}
```

## 📈 Monitoring & Observability

### MLflow Tracking
- Experiment tracking and comparison
- Model versioning and registry
- Hyperparameter optimization tracking
- Access at: http://localhost:5000

### Prometheus Metrics
- `predictions_total` - Total prediction count
- `predictions_by_class` - Predictions per iris class
- `prediction_duration_seconds` - Response time histogram
- Access at: http://localhost:9090

### Grafana Dashboards
- Real-time prediction monitoring
- Performance metrics visualization
- Class distribution analysis
- Access at: http://localhost:3000 (admin/admin)

## 🧪 Testing

```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=src --cov=api --cov-report=html

# Run specific test suite
pytest tests/test_api.py -v

# Quick API test
python scripts/quick_test.py
```

## 🚢 Deployment

### Local Deployment
```bash
docker-compose up -d
```

### Cloud Deployment (AWS EC2)
```bash
# On EC2 instance
./scripts/ec2-setup.sh
docker-compose up -d
```

### CI/CD Pipeline
The GitHub Actions workflow automatically:
1. Runs tests and code quality checks
2. Builds and pushes Docker images
3. Deploys to configured environments

## 🔄 Model Retraining

### Manual Retraining
```bash
python src/retrain.py
```

### Automatic Retraining
```bash
python src/auto_retrain_monitor.py
```

### API Trigger
```bash
curl -X POST http://localhost:8000/retrain
```

## 🛠️ Development

### Setup Development Environment
```bash
# Install development dependencies
pip install -r requirements.txt

# Run code formatting
black src/ api/

# Run linting
flake8 src/ api/

# Run type checking
mypy src/ api/
```

### Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is intended solely for educational purposes and not to use for any other purposes.

## 🙏 Acknowledgments

- Scikit-learn for the Iris dataset
- FastAPI for the excellent web framework
- MLflow for experiment tracking capabilities
- Prometheus and Grafana for monitoring solutions

## 📞 Contact

Project Link: [https://github.com/lokrathod/mlops-iris-pipeline](https://github.com/lokrathod/mlops-iris-pipeline)