"""
Synthetic data generator for Endo Digital Twin.
Creates realistic endometriosis pain data with known relationships.
"""

import pandas as pd
import numpy as np
from typing import Tuple
import os

def make_synth(n: int = 8000) -> pd.DataFrame:
    """
    Generate synthetic endometriosis pain data with realistic relationships.
    
    Args:
        n: Number of samples to generate
        
    Returns:
        DataFrame with features and pain target
    """
    np.random.seed(42)
    
    # Generate features
    data = {}
    
    # Sleep hours: normal distribution around 7, clipped to 3-11
    data['sleep'] = np.clip(np.random.normal(7, 1.5, n), 3, 11)
    
    # Stress level: 0-10 integer
    data['stress'] = np.random.randint(0, 11, n)
    
    # Activity level: 0-10 integer  
    data['activity'] = np.random.randint(0, 11, n)
    
    # Period phase: -2 to +2 integer (0 = period window)
    data['period_phase'] = np.random.randint(-2, 3, n)
    
    # GI symptoms: binary
    data['gi'] = np.random.randint(0, 2, n)
    
    # NSAID medication: binary
    data['meds'] = np.random.randint(0, 2, n)
    
    # Mood: inversely related to stress + noise (0-10)
    mood_base = 10 - data['stress'] + np.random.normal(0, 1.5, n)
    data['mood'] = np.clip(mood_base, 0, 10)
    
    # Hydration: 0-12 glasses per day
    data['hydration'] = np.random.randint(0, 13, n)
    
    # Generate pain target using transparent formula
    pain = (
        4.0  # base pain
        + 0.9 * (data['stress'] / 10)  # stress increases pain
        - 0.45 * ((data['sleep'] - 7) / 3)  # sleep deviation from 7h increases pain
        - 0.2 * (data['activity'] / 10)  # activity decreases pain
        + 0.6 * (data['period_phase'] == 0)  # period window increases pain
        + 0.35 * (np.abs(data['period_phase']) == 1)  # adjacent phases increase pain
        + 0.35 * data['gi']  # GI symptoms increase pain
        - 0.3 * data['meds']  # NSAID decreases pain
        - 0.15 * (data['hydration'] / 10)  # hydration decreases pain
        - 0.25 * (data['mood'] / 10)  # better mood decreases pain
        + np.random.normal(0, 0.6, n)  # noise
    )
    
    # Clip pain to 0-10 range
    data['pain'] = np.clip(pain, 0, 10)
    
    return pd.DataFrame(data)

def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate the generated data meets expected constraints.
    
    Args:
        df: Generated DataFrame
        
    Returns:
        True if validation passes
    """
    # Check pain range
    if not (df['pain'].min() >= 0 and df['pain'].max() <= 10):
        return False
    
    # Check sleep range
    if not (df['sleep'].min() >= 3 and df['sleep'].max() <= 11):
        return False
    
    # Check stress range
    if not (df['stress'].min() >= 0 and df['stress'].max() <= 10):
        return False
    
    # Check activity range
    if not (df['activity'].min() >= 0 and df['activity'].max() <= 10):
        return False
    
    # Check period phase range
    if not (df['period_phase'].min() >= -2 and df['period_phase'].max() <= 2):
        return False
    
    # Check mood range
    if not (df['mood'].min() >= 0 and df['mood'].max() <= 10):
        return False
    
    # Check hydration range
    if not (df['hydration'].min() >= 0 and df['hydration'].max() <= 12):
        return False
    
    # Check binary features
    if not (set(df['gi'].unique()) <= {0, 1}):
        return False
    
    if not (set(df['meds'].unique()) <= {0, 1}):
        return False
    
    return True

if __name__ == "__main__":
    # Generate synthetic data
    print("🩸 Generating synthetic endometriosis data...")
    df = make_synth(8000)
    
    # Validate data
    if validate_data(df):
        print("✅ Data validation passed")
    else:
        print("❌ Data validation failed")
        exit(1)
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save to CSV
    output_path = "data/synthetic.csv"
    df.to_csv(output_path, index=False)
    
    print(f"📊 Generated {len(df)} samples")
    print(f"💾 Saved to {output_path}")
    print(f"📈 Pain range: {df['pain'].min():.2f} - {df['pain'].max():.2f}")
    print(f"📈 Pain mean: {df['pain'].mean():.2f}")
    print(f"📈 Pain std: {df['pain'].std():.2f}")
    
    # Show feature correlations with pain
    print("\n🔍 Feature correlations with pain:")
    correlations = df.corr()['pain'].sort_values(key=abs, ascending=False)
    for feature, corr in correlations.items():
        if feature != 'pain':
            print(f"  {feature:12}: {corr:6.3f}")
