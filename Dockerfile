FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p data/raw data/processed models logs mlruns

# Copy application code
COPY src/ ./src/
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create startup script
RUN echo '#!/bin/bash\n\
echo "Starting MLOps Iris Pipeline..."\n\
\n\
# Check if models exist, if not train them\n\
if [ ! -f "models/best_model.pkl" ]; then\n\
    echo "No models found. Training models..."\n\
    python src/data_preprocessing.py\n\
    python src/train.py\n\
fi\n\
\n\
# Start the API\n\
echo "Starting API server..."\n\
exec uvicorn api.app:app --host 0.0.0.0 --port 8000\n\
' > /app/start.sh && chmod +x /app/start.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["/app/start.sh"]