# New model_ensemble.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.calibration import CalibratedClassifierCV

class XGHACModels:
    def __init__(self):
        self.classifier = CalibratedClassifierCV(
            LogisticRegression(max_iter=1000, class_weight='balanced')
        )
        self.anomaly_detector = IsolationForest(
            n_estimators=200, 
            contamination='auto'
        )
    
    def train_ensemble(self, X, y):
        # Supervised training
        self.classifier.fit(X, y)
        
        # Unsupervised training on successes
        X_success = X[y == 0]
        self.anomaly_detector.fit(X_success)
    
    def predict(self, X):
        proba = self.classifier.predict_proba(X)[:, 1]
        anomalies = self.anomaly_detector.decision_function(X)
        return self._combine_predictions(proba, anomalies)
    
    def _combine_predictions(self, proba, anomalies):
        # Advanced fusion logic
        return 0.7 * proba + 0.3 * (1 - anomalies)