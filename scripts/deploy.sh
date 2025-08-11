#!/bin/bash

set -e

echo "Deploying Iris Classifier API..."

# Check if running in CI or local
if [ -z "$DOCKER_HUB_USERNAME" ]; then
    # Local deployment
    IMAGE_NAME="iris-classifier:latest"
else
    # CI deployment
    IMAGE_NAME="${DOCKER_HUB_USERNAME}/iris-classifier:latest"
    
    # Pull latest image
    docker pull ${IMAGE_NAME}
fi

# Stop existing container if running
docker stop iris-api || true
docker rm iris-api || true

# Run new container
docker run -d \
  --name iris-api \
  -p 8000:8000 \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/models:/app/models \
  --restart unless-stopped \
  ${IMAGE_NAME}

echo "Deployment complete! API available at http://localhost:8000"

# Health check
sleep 5
curl -f http://localhost:8000/ || exit 1
echo "Health check passed!"