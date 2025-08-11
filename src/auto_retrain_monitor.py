"""
Automatic model retraining monitor
"""

import os
import time
import hashlib
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import schedule
from train import train_models

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataChangeHandler(FileSystemEventHandler):
    """Monitor data directory for changes"""
    
    def __init__(self, retrainer):
        self.retrainer = retrainer
        self.last_trigger = datetime.now() - timedelta(minutes=5)  # Cooldown period
        
    def on_modified(self, event):
        if event.src_path.endswith('iris.csv') and not event.is_directory:
            # Avoid multiple triggers in quick succession
            if (datetime.now() - self.last_trigger).seconds > 60:
                logger.info(f"Data file changed: {event.src_path}")
                self.last_trigger = datetime.now()
                self.retrainer.check_and_retrain()

class AutoRetrainer:
    """Automatic model retraining based on various triggers"""
    
    def __init__(self):
        self.data_hash_file = Path('models/data_hash.txt')
        self.last_training_file = Path('models/last_training.txt')
        self.performance_file = Path('models/performance_metrics.txt')
        self.retrain_log = Path('logs/retrain_history.log')
        
        # Ensure directories exist
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
    def get_data_hash(self):
        """Calculate hash of current data"""
        try:
            data = pd.read_csv('data/raw/iris.csv')
            return hashlib.md5(data.to_csv().encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating data hash: {e}")
            return None
    
    def check_data_drift(self):
        """Check for data drift by comparing statistics"""
        try:
            current_data = pd.read_csv('data/raw/iris.csv')
            
            # Calculate basic statistics
            stats = {
                'mean': current_data.select_dtypes(include=[float, int]).mean().to_dict(),
                'std': current_data.select_dtypes(include=[float, int]).std().to_dict(),
                'shape': current_data.shape
            }
            
            # Compare with stored statistics
            stats_file = Path('models/data_stats.json')
            if stats_file.exists():
                import json
                with open(stats_file, 'r') as f:
                    old_stats = json.load(f)
                
                # Simple drift detection: check if mean changed significantly
                for col, mean in stats['mean'].items():
                    if col in old_stats['mean']:
                        drift = abs(mean - old_stats['mean'][col]) / old_stats['mean'][col]
                        if drift > 0.1:  # 10% change
                            return True, f"Drift detected in {col}: {drift:.2%}"
            
            # Save current stats
            import json
            with open(stats_file, 'w') as f:
                json.dump(stats, f)
                
            return False, None
            
        except Exception as e:
            logger.error(f"Error checking data drift: {e}")
            return False, None
    
    def check_model_performance(self):
        """Check if model performance has degraded"""
        try:
            # Read recent predictions log
            predictions_file = Path('logs/predictions.jsonl')
            if not predictions_file.exists():
                return False, None
            
            # Analyze recent predictions
            recent_predictions = []
            with open(predictions_file, 'r') as f:
                for line in f:
                    try:
                        import json
                        recent_predictions.append(json.loads(line))
                    except:
                        continue
            
            if len(recent_predictions) < 100:
                return False, None
            
            # Check average confidence
            recent_confidences = [p['confidence'] for p in recent_predictions[-100:]]
            avg_confidence = sum(recent_confidences) / len(recent_confidences)
            
            # If confidence drops below threshold, trigger retraining
            if avg_confidence < 0.85:
                return True, f"Low average confidence: {avg_confidence:.2f}"
            
            return False, None
            
        except Exception as e:
            logger.error(f"Error checking performance: {e}")
            return False, None
    
    def should_retrain(self):
        """Check all conditions for retraining"""
        reasons = []
        
        # Check 1: Data changed
        current_hash = self.get_data_hash()
        if current_hash and self.data_hash_file.exists():
            stored_hash = self.data_hash_file.read_text().strip()
            if current_hash != stored_hash:
                reasons.append("Data hash changed")
        
        # Check 2: Time-based (weekly)
        if self.last_training_file.exists():
            last_training = datetime.fromisoformat(self.last_training_file.read_text().strip())
            days_since = (datetime.now() - last_training).days
            if days_since >= 7:
                reasons.append(f"Model is {days_since} days old")
        
        # Check 3: Data drift
        has_drift, drift_reason = self.check_data_drift()
        if has_drift:
            reasons.append(drift_reason)
        
        # Check 4: Performance degradation
        has_degraded, perf_reason = self.check_model_performance()
        if has_degraded:
            reasons.append(perf_reason)
        
        return len(reasons) > 0, reasons
    
    def check_and_retrain(self):
        """Check conditions and retrain if needed"""
        should_retrain, reasons = self.should_retrain()
        
        if should_retrain:
            logger.info(f"🔄 Retraining triggered! Reasons: {reasons}")
            
            # Log retrain event
            with open(self.retrain_log, 'a') as f:
                f.write(f"{datetime.now().isoformat()} - Retraining triggered: {reasons}\n")
            
            try:
                # Run training
                logger.info("Starting model retraining...")
                train_models()
                
                # Update tracking files
                current_hash = self.get_data_hash()
                if current_hash:
                    self.data_hash_file.write_text(current_hash)
                
                self.last_training_file.write_text(datetime.now().isoformat())
                
                logger.info("✅ Retraining completed successfully!")
                
                # Log success
                with open(self.retrain_log, 'a') as f:
                    f.write(f"{datetime.now().isoformat()} - Retraining completed successfully\n")
                
            except Exception as e:
                logger.error(f"❌ Retraining failed: {e}")
                with open(self.retrain_log, 'a') as f:
                    f.write(f"{datetime.now().isoformat()} - Retraining failed: {e}\n")
        else:
            logger.info("✓ No retraining needed")

def run_monitoring():
    """Run automatic monitoring"""
    retrainer = AutoRetrainer()
    
    # Set up file system monitoring
    event_handler = DataChangeHandler(retrainer)
    observer = Observer()
    observer.schedule(event_handler, path='data/raw', recursive=False)
    observer.start()
    
    # Set up scheduled checks
    schedule.every(1).hours.do(retrainer.check_and_retrain)
    schedule.every().day.at("02:00").do(retrainer.check_and_retrain)
    
    logger.info("🚀 Automatic retraining monitor started!")
    logger.info("Monitoring:")
    logger.info("  - File changes in data/raw/")
    logger.info("  - Hourly performance checks")
    logger.info("  - Daily scheduled check at 02:00")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(30)  # Check every 30 seconds
    except KeyboardInterrupt:
        observer.stop()
        logger.info("Monitor stopped")
    observer.join()

if __name__ == "__main__":
    # Run initial check
    retrainer = AutoRetrainer()
    retrainer.check_and_retrain()
    
    # Start monitoring
    print("\nStarting automatic monitoring... (Press Ctrl+C to stop)")
    run_monitoring()