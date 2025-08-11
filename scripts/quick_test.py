"""
Quick test script for demo - generates varied predictions
"""

import requests
import random
import time
import json

def test_all_endpoints():
    """Test all API endpoints"""
    
    base_url = "http://localhost:8000"
    
    print("🧪 Testing MLOps Pipeline\n")
    
    # 1. Health Check
    print("1️⃣ Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/")
        print(f"   Status: {response.json()}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print("   Make sure the API is running!")
        return
    
    # 2. Test Predictions
    print("\n2️⃣ Testing Predictions...")
    
    test_samples = {
        'setosa': {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
        'versicolor': {"sepal_length": 6.0, "sepal_width": 2.7, "petal_length": 4.5, "petal_width": 1.5},
        'virginica': {"sepal_length": 6.5, "sepal_width": 3.0, "petal_length": 5.5, "petal_width": 2.0}
    }
    
    for expected, sample in test_samples.items():
        try:
            response = requests.post(f"{base_url}/predict", json=sample)
            result = response.json()
            print(f"   {expected}: {result['prediction_label']} (confidence: {result['confidence']:.3f})")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # 3. Generate Multiple Predictions
    print("\n3️⃣ Generating 30 predictions for monitoring...")
    success_count = 0
    for i in range(30):
        # Randomly select a sample and add noise
        sample = random.choice(list(test_samples.values()))
        noisy_sample = {k: v + random.uniform(-0.1, 0.1) for k, v in sample.items()}
        
        try:
            response = requests.post(f"{base_url}/predict", json=noisy_sample)
            if response.status_code == 200:
                success_count += 1
            if (i + 1) % 10 == 0:
                print(f"   Progress: {i+1}/30")
        except:
            pass
        time.sleep(0.1)
    
    print(f"   ✅ Successfully made {success_count}/30 predictions")
    
    # 4. Check Metrics
    print("\n4️⃣ Checking Prometheus Metrics...")
    try:
        response = requests.get(f"{base_url}/metrics")
        if response.status_code == 200:
            metrics_lines = response.text.split('\n')
            for line in metrics_lines:
                if 'predictions_total' in line and not line.startswith('#'):
                    print(f"   {line}")
                    break
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 5. View Logs
    print("\n5️⃣ Recent Predictions Log:")
    try:
        with open('logs/predictions.jsonl', 'r') as f:
            lines = f.readlines()
            if lines:
                for line in lines[-3:]:  # Last 3 predictions
                    log = json.loads(line)
                    print(f"   {log['timestamp']}: {log['prediction_label']} (confidence: {log['confidence']:.3f})")
            else:
                print("   No predictions logged yet")
    except FileNotFoundError:
        print("   Log file not found yet")
    except Exception as e:
        print(f"   Error reading logs: {e}")
    
    print("\n✅ All tests completed!")
    print("\n📊 Check these URLs:")
    print("   - API Docs: http://localhost:8000/docs")
    print("   - MLflow: http://localhost:5000")
    print("   - Prometheus: http://localhost:9090")
    print("   - Grafana: http://localhost:3000")

def generate_varied_predictions(n=100):
    """Generate predictions for all three classes"""
    
    base_url = "http://localhost:8000"
    
    # Samples designed to predict each class
    samples = {
        'setosa': [
            {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
            {"sepal_length": 4.9, "sepal_width": 3.0, "petal_length": 1.4, "petal_width": 0.2},
            {"sepal_length": 4.7, "sepal_width": 3.2, "petal_length": 1.3, "petal_width": 0.2},
        ],
        'versicolor': [
            {"sepal_length": 6.0, "sepal_width": 2.7, "petal_length": 4.5, "petal_width": 1.5},
            {"sepal_length": 5.9, "sepal_width": 3.0, "petal_length": 4.2, "petal_width": 1.5},
            {"sepal_length": 6.2, "sepal_width": 2.2, "petal_length": 4.5, "petal_width": 1.5},
        ],
        'virginica': [
            {"sepal_length": 6.5, "sepal_width": 3.0, "petal_length": 5.5, "petal_width": 2.0},
            {"sepal_length": 7.2, "sepal_width": 3.6, "petal_length": 6.1, "petal_width": 2.5},
            {"sepal_length": 6.9, "sepal_width": 3.1, "petal_length": 5.4, "petal_width": 2.1},
        ]
    }
    
    print(f"\n🚀 Generating {n} varied predictions...")
    counts = {'setosa': 0, 'versicolor': 0, 'virginica': 0}
    
    for i in range(n):
        # Rotate through classes to ensure variety
        class_name = list(samples.keys())[i % 3]
        sample = random.choice(samples[class_name])
        
        # Add small random noise
        noisy_sample = {
            k: v + random.uniform(-0.05, 0.05) for k, v in sample.items()
        }
        
        try:
            response = requests.post(f"{base_url}/predict", json=noisy_sample, timeout=2)
            
            if response.status_code == 200:
                result = response.json()
                predicted_class = result['prediction_label']
                counts[predicted_class] = counts.get(predicted_class, 0) + 1
                
                if (i + 1) % 20 == 0:
                    print(f"   Progress: {i+1}/{n} - Last prediction: {predicted_class}")
                    
        except Exception as e:
            if i == 0:
                print(f"   ❌ Error: {e}")
                print("   Make sure the API is running!")
                return
        
        # Small delay to spread out the requests
        time.sleep(0.05)
    
    print(f"\n✅ Generated {n} predictions!")
    print(f"Distribution: {counts}")
    print("\n🎉 Check your Grafana dashboard now!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--varied":
        # Generate varied predictions for better dashboard
        generate_varied_predictions(100)
    else:
        # Run basic tests
        test_all_endpoints()