#!/usr/bin/env python3
# src/retrain.py
"""
Manual model retraining script
"""

import pandas as pd
import hashlib
from pathlib import Path
import logging
import time
from datetime import datetime
from train import train_models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelRetrainer:
    def __init__(self):
        self.data_hash_file = Path('models/data_hash.txt')
        self.last_training_file = Path('models/last_training.txt')
        
    def get_data_hash(self):
        """Calculate hash of current data"""
        try:
            data = pd.read_csv('data/raw/iris.csv')
            return hashlib.md5(data.to_csv().encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating data hash: {e}")
            return None
    
    def should_retrain(self):
        """Check if retraining is needed"""
        reasons = []
        
        # Check 1: Data changed
        current_hash = self.get_data_hash()
        if current_hash and self.data_hash_file.exists():
            stored_hash = self.data_hash_file.read_text().strip()
            if current_hash != stored_hash:
                reasons.append("Data has changed")
        
        # Check 2: Time-based (e.g., weekly retraining)
        if self.last_training_file.exists():
            last_training = datetime.fromisoformat(self.last_training_file.read_text().strip())
            days_since_training = (datetime.now() - last_training).days
            if days_since_training >= 7:
                reasons.append(f"Model is {days_since_training} days old")
        
        return len(reasons) > 0, reasons
    
    def retrain(self):
        """Trigger model retraining"""
        should_retrain, reasons = self.should_retrain()
        
        if should_retrain:
            logger.info(f"Retraining triggered. Reasons: {reasons}")
            
            try:
                # Run training
                train_models()
                
                # Update tracking files
                current_hash = self.get_data_hash()
                if current_hash:
                    self.data_hash_file.write_text(current_hash)
                
                self.last_training_file.write_text(datetime.now().isoformat())
                
                logger.info("Retraining completed successfully")
                
            except Exception as e:
                logger.error(f"Retraining failed: {e}")
        else:
            logger.info("No retraining needed")

def check_and_retrain():
    """Function to be scheduled"""
    retrainer = ModelRetrainer()
    retrainer.retrain()

if __name__ == "__main__":
    # Run once
    check_and_retrain()