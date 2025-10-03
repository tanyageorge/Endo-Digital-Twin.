"""
Explanation module for ElasticNet predictions.
Provides natural language explanations based on model coefficients.
"""

from typing import Dict, Any, List, Tuple

def delta_explanation(baseline: Dict[str, Any], scenario: Dict[str, Any], 
                     coefficients: Dict[str, float]) -> str:
    """
    Generate natural language explanation for pain prediction change.
    
    Args:
        baseline: Baseline feature values
        scenario: Scenario feature values  
        coefficients: Model coefficients from ElasticNet
        
    Returns:
        Natural language explanation string
    """
    # Calculate changes for each feature
    changes = {}
    for feature in coefficients.keys():
        if feature in baseline and feature in scenario:
            changes[feature] = scenario[feature] - baseline[feature]
    
    # Calculate contribution of each feature to pain change
    contributions = {}
    for feature, change in changes.items():
        if feature in coefficients:
            # Contribution = change * coefficient
            contributions[feature] = change * coefficients[feature]
    
    # Sort by absolute contribution
    sorted_contributions = sorted(contributions.items(), 
                                key=lambda x: abs(x[1]), reverse=True)
    
    # Get top 2-3 contributors
    top_contributors = sorted_contributions[:3]
    
    # Filter out very small contributions
    significant_contributors = [(feat, contrib) for feat, contrib in top_contributors 
                              if abs(contrib) > 0.05]
    
    if not significant_contributors:
        return "The changes have minimal impact on pain prediction."
    
    # Build explanation
    explanation_parts = []
    
    for i, (feature, contribution) in enumerate(significant_contributors):
        if i == 0:
            # First contributor
            if contribution > 0:
                explanation_parts.append(f"increased {feature} increases pain")
            else:
                explanation_parts.append(f"decreased {feature} decreases pain")
        else:
            # Additional contributors
            if contribution > 0:
                explanation_parts.append(f"increased {feature} increases pain")
            else:
                explanation_parts.append(f"decreased {feature} decreases pain")
    
    # Join with appropriate conjunctions
    if len(explanation_parts) == 1:
        explanation = f"The {explanation_parts[0]}."
    elif len(explanation_parts) == 2:
        explanation = f"The {explanation_parts[0]} and {explanation_parts[1]}."
    else:
        explanation = f"The {explanation_parts[0]}, {explanation_parts[1]}, and {explanation_parts[2]}."
    
    return explanation

def format_feature_change(feature: str, baseline: float, scenario: float) -> str:
    """
    Format feature change in a readable way.
    
    Args:
        feature: Feature name
        baseline: Baseline value
        scenario: Scenario value
        
    Returns:
        Formatted change string
    """
    change = scenario - baseline
    
    if abs(change) < 0.01:
        return f"{feature} unchanged"
    
    if change > 0:
        return f"{feature} +{change:.1f}"
    else:
        return f"{feature} {change:.1f}"

def get_feature_descriptions() -> Dict[str, str]:
    """
    Get human-readable descriptions for features.
    
    Returns:
        Dictionary mapping feature names to descriptions
    """
    return {
        'sleep': 'sleep hours',
        'stress': 'stress level',
        'activity': 'activity level', 
        'period_phase': 'menstrual cycle phase',
        'gi': 'GI symptoms',
        'meds': 'NSAID medication',
        'mood': 'mood level',
        'hydration': 'hydration level'
    }

def explain_prediction_change(baseline: Dict[str, Any], scenario: Dict[str, Any],
                            coefficients: Dict[str, float], 
                            pain_change: float) -> str:
    """
    Generate comprehensive explanation for prediction change.
    
    Args:
        baseline: Baseline feature values
        scenario: Scenario feature values
        coefficients: Model coefficients
        pain_change: Change in predicted pain (scenario - baseline)
        
    Returns:
        Comprehensive explanation string
    """
    # Get feature descriptions
    descriptions = get_feature_descriptions()
    
    # Calculate changes
    changes = {}
    for feature in coefficients.keys():
        if feature in baseline and feature in scenario:
            changes[feature] = scenario[feature] - baseline[feature]
    
    # Find significant changes (threshold = 0.1 for continuous, 1 for binary)
    significant_changes = []
    for feature, change in changes.items():
        if feature in ['gi', 'meds', 'period_phase']:
            # Binary or categorical features
            if abs(change) >= 1:
                significant_changes.append((feature, change))
        else:
            # Continuous features
            if abs(change) >= 0.1:
                significant_changes.append((feature, change))
    
    if not significant_changes:
        return "No significant changes in features."
    
    # Sort by absolute change
    significant_changes.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Build explanation
    change_parts = []
    for feature, change in significant_changes[:3]:  # Top 3 changes
        desc = descriptions.get(feature, feature)
        if change > 0:
            change_parts.append(f"increased {desc}")
        else:
            change_parts.append(f"decreased {desc}")
    
    # Join changes
    if len(change_parts) == 1:
        changes_text = change_parts[0]
    elif len(change_parts) == 2:
        changes_text = f"{change_parts[0]} and {change_parts[1]}"
    else:
        changes_text = f"{change_parts[0]}, {change_parts[1]}, and {change_parts[2]}"
    
    # Add pain change context
    if pain_change > 0.1:
        direction = "increases"
    elif pain_change < -0.1:
        direction = "decreases"
    else:
        direction = "has minimal effect on"
    
    explanation = f"Changes in {changes_text} {direction} predicted pain by {pain_change:+.1f} points."
    
    return explanation

if __name__ == "__main__":
    # Test explanation functions
    print("🧠 Testing explanation module...")
    
    # Test data
    baseline = {
        'sleep': 8.0,
        'stress': 5,
        'activity': 5,
        'period_phase': 0,
        'gi': 0,
        'meds': 0,
        'mood': 7,
        'hydration': 8
    }
    
    scenario = {
        'sleep': 6.0,  # Less sleep
        'stress': 8,   # More stress
        'activity': 3, # Less activity
        'period_phase': 0,
        'gi': 1,       # GI symptoms
        'meds': 0,
        'mood': 4,     # Worse mood
        'hydration': 6 # Less hydration
    }
    
    # Mock coefficients (similar to what ElasticNet would produce)
    coefficients = {
        'sleep': -0.3,
        'stress': 0.8,
        'activity': -0.2,
        'period_phase': 0.6,
        'gi': 0.4,
        'meds': -0.3,
        'mood': -0.25,
        'hydration': -0.15
    }
    
    # Test explanation
    explanation = delta_explanation(baseline, scenario, coefficients)
    print(f"\n📝 Delta explanation: {explanation}")
    
    # Test comprehensive explanation
    pain_change = 2.1  # Mock pain change
    comprehensive = explain_prediction_change(baseline, scenario, coefficients, pain_change)
    print(f"\n📊 Comprehensive explanation: {comprehensive}")
    
    print("\n✅ Explanation module test completed!")
