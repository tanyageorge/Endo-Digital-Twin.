"""
ElasticNet training pipeline for Endo Digital Twin.
Trains a linear model with L1+L2 regularization for interpretable predictions.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
import os
from typing import Dict, Any, Tuple

# Feature order (must match across all models)
FEATURES = ["sleep", "stress", "activity", "period_phase", "gi", "meds", "mood", "hydration"]

def load_data(data_path: str = "data/synthetic.csv") -> Tuple[pd.DataFrame, pd.Series]:
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

def train_elasticnet(X: pd.DataFrame, y: pd.Series, 
                    test_size: float = 0.2, 
                    random_state: int = 42) -> Dict[str, Any]:
    """
    Train ElasticNet model with cross-validation and evaluation.
    
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
    
    # Create pipeline: StandardScaler + ElasticNet
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('elasticnet', ElasticNet(
            alpha=0.12,  # Regularization strength
            l1_ratio=0.2,  # Mix of L1/L2 (0.2 = mostly L2)
            max_iter=5000,
            random_state=random_state
        ))
    ])
    
    # Train model
    print("🏋️ Training ElasticNet model...")
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Calculate prediction interval half-width from residuals
    test_residuals = np.abs(y_test - y_test_pred)
    pi_halfwidth = np.percentile(test_residuals, 80)  # 80th percentile
    
    # Extract coefficients for interpretation
    coefficients = {}
    feature_names = FEATURES
    coef_values = pipeline.named_steps['elasticnet'].coef_
    
    for feature, coef in zip(feature_names, coef_values):
        coefficients[feature] = float(coef)
    
    # Print results
    print("\n📊 ElasticNet Results:")
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²:  {test_r2:.4f}")
    print(f"  Train MAE: {train_mae:.4f}")
    print(f"  Test MAE:  {test_mae:.4f}")
    print(f"  PI Half-width: {pi_halfwidth:.4f}")
    
    print("\n🔍 Feature Coefficients:")
    for feature, coef in sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True):
        direction = "increases" if coef > 0 else "decreases"
        print(f"  {feature:12}: {coef:7.4f} ({direction} pain)")
    
    return {
        'pipeline': pipeline,
        'features': FEATURES,
        'meta': {
            'type': 'elasticnet',
            'r2': float(test_r2),
            'mae': float(test_mae),
            'pi_halfwidth': float(pi_halfwidth),
            'coefficients': coefficients,
            'train_r2': float(train_r2),
            'train_mae': float(train_mae)
        }
    }

def save_model(model_data: Dict[str, Any], output_path: str = "models/model_en.pkl") -> None:
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
    print("🩸 Training ElasticNet for Endo Digital Twin")
    print("=" * 50)
    
    # Load data
    X, y = load_data()
    
    # Train model
    model_data = train_elasticnet(X, y)
    
    # Save model
    save_model(model_data)
    
    print("\n✅ ElasticNet training completed successfully!")

if __name__ == "__main__":
    main()
