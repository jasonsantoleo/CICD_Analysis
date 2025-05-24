# Enhanced feature_engineer.py
from transformers import AutoTokenizer, AutoModel
import numpy as np

class LogProcessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")
    
    def process_logs(self, logs):
        """Enhanced log processing with BERT embeddings"""
        encoded = self.tokenizer(
            logs["message"].tolist(),
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        outputs = self.model(**encoded)
        embeddings = np.mean(outputs.last_hidden_state.detach().numpy(), axis=1)
        return embeddings

class MetricAnalyzer:
    def extract_features(self, metrics):
        """Time-series feature engineering"""
        return {
            "cpu_5min_avg": self._rolling_average(metrics["cpu"], window=5),
            "memory_spike": self._detect_spikes(metrics["memory"]),
            "disk_trend": self._calculate_trend(metrics["disk"])
        }
    
    # Implement statistical methods...