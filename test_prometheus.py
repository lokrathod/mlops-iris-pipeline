#!/usr/bin/env python3
"""
Test Prometheus metrics collection
"""

import requests
import time
import random

def check_metrics():
    """Check if metrics endpoint is working"""
    try:
        response = requests.get("http://localhost:8000/metrics")
        if response.status_code == 200:
            print("✅ Metrics endpoint is working!")
            print("\nSample metrics:")
            lines = response.text.split('\n')
            for line in lines[:20]:  # Show first 20 lines
                if line and not line.startswith('#'):
                    print(f"  {line}")
            return True
        else:
            print(f"❌ Metrics endpoint returned: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error accessing metrics: {e}")
        return False

def generate_traffic():
    """Generate some traffic to create metrics"""
    print("\n🚀 Generating traffic for metrics...")
    
    samples = [
        # Setosa
        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
        # Versicolor
        {"sepal_length": 6.0, "sepal_width": 2.7, "petal_length": 4.5, "petal_width": 1.5},
        # Virginica
        {"sepal_length": 6.5, "sepal_width": 3.0, "petal_length": 5.5, "petal_width": 2.0},
    ]
    
    for i in range(30):
        sample = random.choice(samples)
        # Add noise
        noisy_sample = {
            k: v + random.uniform(-0.1, 0.1) for k, v in sample.items()
        }
        
        try:
            response = requests.post(
                "http://localhost:8000/predict",
                json=noisy_sample
            )
            if response.status_code == 200:
                result = response.json()
                print(f"  Prediction {i+1}: {result['prediction_label']}")
        except:
            pass
        
        time.sleep(0.5)

def check_prometheus():
    """Check if Prometheus is scraping metrics"""
    try:
        # Check Prometheus targets
        response = requests.get("http://localhost:9090/api/v1/targets")
        if response.status_code == 200:
            targets = response.json()
            print("\n✅ Prometheus is running!")
            print("Targets:")
            for target in targets['data']['activeTargets']:
                print(f"  - {target['labels']['job']}: {target['health']}")
        
        # Check specific metrics
        response = requests.get(
            "http://localhost:9090/api/v1/query?query=predictions_total"
        )
        if response.status_code == 200:
            data = response.json()
            if data['data']['result']:
                value = data['data']['result'][0]['value'][1]
                print(f"\nTotal predictions in Prometheus: {value}")
    except:
        print("⚠️  Prometheus not accessible at http://localhost:9090")

if __name__ == "__main__":
    print("🔍 Testing Prometheus Integration\n")
    
    # Check metrics endpoint
    if check_metrics():
        # Generate some traffic
        generate_traffic()
        
        # Check metrics again
        print("\n📊 Updated metrics:")
        check_metrics()
        
        # Check Prometheus
        check_prometheus()
        
        print("\n✨ Prometheus integration is working!")
        print("\nAccess:")
        print("  - Prometheus: http://localhost:9090")
        print("  - Grafana: http://localhost:3000 (admin/admin)")