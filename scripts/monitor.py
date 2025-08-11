import json
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def analyze_predictions(log_file='logs/predictions.jsonl', last_hours=24):
    """Analyze prediction logs"""
    
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found. Make some predictions first!")
        return
    
    # Read logs
    predictions = []
    with open(log_file, 'r') as f:
        for line in f:
            try:
                predictions.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    if not predictions:
        print("No predictions found in log file.")
        return
    
    df = pd.DataFrame(predictions)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter by time
    cutoff_time = datetime.utcnow() - timedelta(hours=last_hours)
    df = df[df['timestamp'] > cutoff_time]
    
    print(f"\n=== Prediction Analytics (Last {last_hours} hours) ===")
    print(f"Total predictions: {len(df)}")
    
    if len(df) == 0:
        print("No predictions in the specified time range.")
        return
    
    print(f"\nPredictions by class:")
    print(df['prediction_label'].value_counts())
    
    print(f"\nAverage confidence: {df['confidence'].mean():.3f}")
    print(f"Min confidence: {df['confidence'].min():.3f}")
    print(f"Max confidence: {df['confidence'].max():.3f}")
    
    print(f"\nAverage response time: {df['duration'].mean():.3f}s")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Predictions over time
    if len(df) > 1:
        df.set_index('timestamp').resample('1H')['prediction'].count().plot(
            ax=axes[0, 0]
        )
        axes[0, 0].set_title('Predictions per Hour')
    else:
        axes[0, 0].text(0.5, 0.5, 'Not enough data for time series', 
                        ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Predictions per Hour')
    
    # Class distribution
    df['prediction_label'].value_counts().plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title('Class Distribution')
    axes[0, 1].set_xlabel('Class')
    axes[0, 1].set_ylabel('Count')
    
    # Confidence distribution
    if df['confidence'].nunique() > 1:
        df['confidence'].hist(bins=20, ax=axes[1, 0])
    else:
        axes[1, 0].bar([df['confidence'].iloc[0]], [len(df)], width=0.01)
        axes[1, 0].set_xlim(df['confidence'].iloc[0] - 0.1, df['confidence'].iloc[0] + 0.1)
    axes[1, 0].set_title('Confidence Distribution')
    axes[1, 0].set_xlabel('Confidence')
    axes[1, 0].set_ylabel('Frequency')
    
    # Response time distribution
    if df['duration'].nunique() > 1:
        df['duration'].hist(bins=20, ax=axes[1, 1])
    else:
        axes[1, 1].bar([df['duration'].mean()], [len(df)], width=df['duration'].std() if df['duration'].std() > 0 else 0.001)
    axes[1, 1].set_title('Response Time Distribution')
    axes[1, 1].set_xlabel('Duration (seconds)')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    plt.savefig('logs/monitoring_report.png', dpi=150, bbox_inches='tight')
    print("\nMonitoring report saved to logs/monitoring_report.png")

if __name__ == "__main__":
    analyze_predictions()