#!/bin/bash
# EC2 setup script for MLOps pipeline

set -e

echo "🚀 Setting up EC2 instance for MLOps Pipeline"

# Update system
echo "📦 Updating system packages..."
sudo yum update -y || sudo apt-get update -y

# Install Docker
echo "🐳 Installing Docker..."
if command -v yum &> /dev/null; then
    # Amazon Linux 2
    sudo yum install -y docker git
    sudo service docker start
    sudo usermod -a -G docker $USER
else
    # Ubuntu
    sudo apt-get install -y docker.io docker-compose git
    sudo systemctl start docker
    sudo usermod -a -G docker $USER
fi

# Install Docker Compose
echo "🐳 Installing Docker Compose..."
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Create project directory
echo "📁 Creating project directory..."
mkdir -p ~/mlops-iris-pipeline

# Setup firewall rules (if ufw is available)
if command -v ufw &> /dev/null; then
    echo "🔥 Configuring firewall..."
    sudo ufw allow 22/tcp    # SSH
    sudo ufw allow 80/tcp    # HTTP
    sudo ufw allow 8000/tcp  # API
    sudo ufw allow 5000/tcp  # MLflow
    sudo ufw allow 3000/tcp  # Grafana
    sudo ufw allow 9090/tcp  # Prometheus
    sudo ufw --force enable
fi

echo "✅ EC2 setup complete!"
echo ""
echo "Next steps:"
echo "1. Log out and back in for Docker permissions"
echo "2. Clone repository:"
echo "   git clone https://github.com/YOUR_USERNAME/mlops-iris-pipeline.git"
echo "3. Navigate to project:"
echo "   cd mlops-iris-pipeline"
echo "4. Start services:"
echo "   docker-compose up -d"
echo ""
echo "🎉 Setup complete!"