"""
Model registry and loader for Endo Digital Twin.
Provides uniform interface for loading and using trained models.
"""

import joblib
import numpy as np
from typing import Dict, Any, Optional
import os

class Predictor:
    """
    Wrapper class for trained models with uniform prediction interface.
    """
    
    def __init__(self, path: str, feature_order: list[str]):
        """
        Initialize predictor by loading model from file.
        
        Args:
            path: Path to saved model file
            feature_order: List of feature names in correct order
        """
        self.path = path
        self.feature_order = feature_order
        self.pipeline = None
        self.features = None
        self.meta = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load model from file."""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Model file not found: {self.path}")
        
        try:
            model_data = joblib.load(self.path)
            self.pipeline = model_data['pipeline']
            self.features = model_data['features']
            self.meta = model_data['meta']
            
            # Validate feature order matches
            if self.features != self.feature_order:
                raise ValueError(f"Feature order mismatch: expected {self.feature_order}, got {self.features}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.path}: {str(e)}")
    
    def predict(self, feats_dict: Dict[str, Any]) -> float:
        """
        Make prediction from feature dictionary.
        
        Args:
            feats_dict: Dictionary with feature names as keys and values
            
        Returns:
            Predicted pain level (0-10)
        """
        # Convert to array in correct feature order
        X = np.array([[feats_dict.get(feature, 0) for feature in self.feature_order]])
        
        # Make prediction
        prediction = self.pipeline.predict(X)[0]
        
        # Clip to 0-10 range
        return float(np.clip(prediction, 0, 10))
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        return self.meta.copy() if self.meta else {}
    
    def get_feature_order(self) -> list[str]:
        """Get feature order."""
        return self.feature_order.copy()

def load_predictor(model_key: str) -> Predictor:
    """
    Load predictor by model key.
    
    Args:
        model_key: Model identifier ("ElasticNet" or "RandomForest")
        
    Returns:
        Loaded Predictor instance
        
    Raises:
        ValueError: If model_key is not recognized
        FileNotFoundError: If model file doesn't exist
    """
    # Model key to file path mapping
    model_paths = {
        "ElasticNet": "models/model_en.pkl",
        "RandomForest": "models/model_rf.pkl"
    }
    
    if model_key not in model_paths:
        available = ", ".join(model_paths.keys())
        raise ValueError(f"Unknown model key '{model_key}'. Available: {available}")
    
    path = model_paths[model_key]
    
    # Feature order (must match training)
    feature_order = ["sleep", "stress", "activity", "period_phase", "gi", "meds", "mood", "hydration"]
    
    return Predictor(path, feature_order)

def list_available_models() -> list[str]:
    """
    List available model keys.
    
    Returns:
        List of available model keys
    """
    return ["ElasticNet", "RandomForest"]

def check_model_availability() -> Dict[str, bool]:
    """
    Check which models are available on disk.
    
    Returns:
        Dictionary mapping model keys to availability status
    """
    model_paths = {
        "ElasticNet": "models/model_en.pkl",
        "RandomForest": "models/model_rf.pkl"
    }
    
    availability = {}
    for model_key, path in model_paths.items():
        availability[model_key] = os.path.exists(path)
    
    return availability

def get_model_info(model_key: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a specific model without loading it.
    
    Args:
        model_key: Model identifier
        
    Returns:
        Model metadata or None if not available
    """
    try:
        predictor = load_predictor(model_key)
        return predictor.get_metadata()
    except (FileNotFoundError, ValueError):
        return None

if __name__ == "__main__":
    # Test the registry
    print("🔍 Testing model registry...")
    
    # Check availability
    availability = check_model_availability()
    print("\n📊 Model availability:")
    for model_key, available in availability.items():
        status = "✅" if available else "❌"
        print(f"  {model_key}: {status}")
    
    # Test loading available models
    for model_key in list_available_models():
        if availability[model_key]:
            try:
                predictor = load_predictor(model_key)
                meta = predictor.get_metadata()
                print(f"\n📋 {model_key} info:")
                print(f"  Type: {meta.get('type', 'unknown')}")
                print(f"  R²: {meta.get('r2', 'N/A'):.4f}")
                print(f"  MAE: {meta.get('mae', 'N/A'):.4f}")
                print(f"  PI Half-width: {meta.get('pi_halfwidth', 'N/A'):.4f}")
                
                # Test prediction
                test_features = {
                    'sleep': 8.0,
                    'stress': 5,
                    'activity': 5,
                    'period_phase': 0,
                    'gi': 0,
                    'meds': 0,
                    'mood': 7,
                    'hydration': 8
                }
                prediction = predictor.predict(test_features)
                print(f"  Test prediction: {prediction:.2f}/10")
                
            except Exception as e:
                print(f"❌ Error loading {model_key}: {e}")
        else:
            print(f"⚠️  {model_key} not available")
