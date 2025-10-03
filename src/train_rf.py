"""
RandomForest training pipeline for Endo Digital Twin.
Trains a non-linear ensemble model for complex pattern recognition.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import os
from typing import Dict, Any, Tuple

# Feature order (must match across all models)
FEATURES = ["sleep", "stress", "activity", "period_phase", "gi", "meds", "mood", "hydration"]

def load_data(data_path: str = "data/synthetic.csv") -> tuple[pd.DataFrame, pd.Series]:
    """
    Load and prepare data for training.
    
    Args:
        data_path: Path to synthetic data CSV
        
    Returns:
        X: Feature matrix
        y: Target vector
    """
    df = pd.read_csv(data_path)
    
    # Ensure features are in correct order
    X = df[FEATURES].copy()
    y = df['pain'].copy()
    
    print(f"📊 Loaded {len(X)} samples with {len(FEATURES)} features")
    print(f"📈 Target range: {y.min():.2f} - {y.max():.2f}")
    
    return X, y

def train_randomforest(X: pd.DataFrame, y: pd.Series, 
                      test_size: float = 0.2, 
                      random_state: int = 42) -> Dict[str, Any]:
    """
    Train RandomForest model with evaluation.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Fraction of data for testing
        random_state: Random seed
        
    Returns:
        Dictionary with model, metrics, and metadata
    """
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"🔄 Train/test split: {len(X_train)}/{len(X_test)} samples")
    
    # Create RandomForest model
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,  # Use all available cores
        max_depth=10,  # Prevent overfitting
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    # Train model
    print("🏋️ Training RandomForest model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Calculate prediction interval half-width from residuals
    test_residuals = np.abs(y_test - y_test_pred)
    pi_halfwidth = np.percentile(test_residuals, 80)  # 80th percentile
    
    # Extract feature importances
    importances = {}
    feature_names = FEATURES
    importance_values = model.feature_importances_
    
    for feature, importance in zip(feature_names, importance_values):
        importances[feature] = float(importance)
    
    # Print results
    print("\n📊 RandomForest Results:")
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²:  {test_r2:.4f}")
    print(f"  Train MAE: {train_mae:.4f}")
    print(f"  Test MAE:  {test_mae:.4f}")
    print(f"  PI Half-width: {pi_halfwidth:.4f}")
    
    print("\n🔍 Feature Importances:")
    for feature, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature:12}: {importance:.4f}")
    
    return {
        'pipeline': model,  # RandomForest is already a pipeline-like object
        'features': FEATURES,
        'meta': {
            'type': 'rf',
            'r2': float(test_r2),
            'mae': float(test_mae),
            'pi_halfwidth': float(pi_halfwidth),
            'feature_importances': importances,
            'train_r2': float(train_r2),
            'train_mae': float(train_mae)
        }
    }

def save_model(model_data: Dict[str, Any], output_path: str = "models/model_rf.pkl") -> None:
    """
    Save trained model with metadata.
    
    Args:
        model_data: Dictionary with model and metadata
        output_path: Path to save the model
    """
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save model
    joblib.dump(model_data, output_path)
    print(f"💾 Model saved to {output_path}")

def main():
    """Main training function."""
    print("🩸 Training RandomForest for Endo Digital Twin")
    print("=" * 50)
    
    # Load data
    X, y = load_data()
    
    # Train model
    model_data = train_randomforest(X, y)
    
    # Save model
    save_model(model_data)
    
    print("\n✅ RandomForest training completed successfully!")

if __name__ == "__main__":
    main()
