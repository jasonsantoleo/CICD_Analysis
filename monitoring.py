# New monitoring.py
from prometheus_client import start_http_server, Gauge
from tinydb import TinyDB
from datetime import datetime
import time

class XGHACMonitor:
    def __init__(self):
        self.prediction_latency = Gauge(
            'xghac_prediction_latency', 
            'Time taken for predictions'
        )
        self.model_drift = Gauge(
            'xghac_model_drift', 
            'KL divergence from reference distribution'
        )
        
    def track_metrics(self):
        start_http_server(8000)
        while True:
            self._update_latency()
            self._check_model_drift()
            time.sleep(60)

class FeedbackHandler:
    def __init__(self):
        self.feedback_db = TinyDB('feedback.json')
    
    def record_feedback(self, prediction_id, is_correct):
        self.feedback_db.insert({
            'timestamp': datetime.now(),
            'prediction_id': prediction_id,
            'valid': is_correct
        })