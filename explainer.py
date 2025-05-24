# New explainer.py
import shap
from typing import Dict, Any

class PredictionExplainer:
    def __init__(self, model, preprocessor):
        self.explainer = shap.Explainer(model, preprocessor)
        self.feature_names = preprocessor.get_feature_names_out()
    
    def explain(self, X) -> Dict[str, Any]:
        shap_values = self.explainer(X)
        return {
            "base_value": float(shap_values.base_values),
            "values": shap_values.values.tolist(),
            "feature_names": self.feature_names,
            "data": X.values.tolist()
        }
    
    def format_for_github(self, explanation):
        """Convert SHAP output to GitHub Check Run format"""
        return "\n".join(
            f"🔍 {name}: {value:.2f}"
            for name, value in zip(
                explanation["feature_names"], 
                explanation["values"]
            )
        )