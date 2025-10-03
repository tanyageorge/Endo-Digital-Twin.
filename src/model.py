"""
Mock prediction model for Endo Digital Twin.
This will be replaced with a trained machine learning model in the future.
"""

import numpy as np
from typing import Dict, Any

class EndoPredictionModel:
    """
    Mock prediction model for endometriosis pain prediction.
    
    This is a simplified model that will be replaced with a trained
    machine learning model in future iterations.
    """
    
    def __init__(self):
        self.model_name = "Mock Endo Prediction Model v0.1"
    
    def predict_pain(self, features: Dict[str, Any]) -> float:
        """
        Predict pain level based on input features.
        
        Args:
            features: Dictionary containing feature values
                - sleep_hours: float (0-12)
                - stress_level: int (0-10)
                - activity_level: int (0-10)
                - hydration_level: int (0-10)
                - period_phase: str (cycle phase)
                - nsaid_taken: bool
                - gi_symptoms: str ("Yes"/"No")
        
        Returns:
            float: Predicted pain level (0-10)
        """
        # Convert period phase to numeric
        period_mapping = {
            "-2 (Ovulation)": -2,
            "-1 (Pre-menstrual)": -1,
            "0 (Menstrual)": 0,
            "+1 (Post-menstrual)": 1,
            "+2 (Follicular)": 2
        }
        
        period_numeric = period_mapping.get(features.get('period_phase', '0 (Menstrual)'), 0)
        
        # Base pain level
        base_pain = 3.0
        
        # Sleep impact (optimal around 8 hours)
        sleep_hours = features.get('sleep_hours', 8.0)
        sleep_impact = abs(sleep_hours - 8) * 0.2
        
        # Stress impact
        stress_level = features.get('stress_level', 5)
        stress_impact = stress_level * 0.3
        
        # Activity impact (both too low and too high can increase pain)
        activity_level = features.get('activity_level', 5)
        activity_impact = abs(activity_level - 5) * 0.15
        
        # Hydration impact
        hydration_level = features.get('hydration_level', 7)
        hydration_impact = (10 - hydration_level) * 0.1
        
        # Period phase impact (menstrual phase increases pain)
        period_impact = abs(period_numeric) * 0.4
        
        # Medication impact
        nsaid_taken = features.get('nsaid_taken', False)
        med_impact = -1.0 if nsaid_taken else 0.0
        
        # GI symptoms impact
        gi_symptoms = features.get('gi_symptoms', 'No')
        gi_impact = 0.5 if gi_symptoms == 'Yes' else 0.0
        
        # Calculate final prediction
        predicted_pain = (base_pain + sleep_impact + stress_impact + 
                         activity_impact + hydration_impact + period_impact + 
                         med_impact + gi_impact)
        
        # Ensure pain is within 0-10 range
        return max(0.0, min(10.0, predicted_pain))
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores (mock values for now).
        
        Returns:
            Dict[str, float]: Feature importance scores
        """
        return {
            'period_phase': 0.25,
            'stress_level': 0.20,
            'sleep_hours': 0.15,
            'activity_level': 0.15,
            'nsaid_taken': 0.10,
            'hydration_level': 0.10,
            'gi_symptoms': 0.05
        }

