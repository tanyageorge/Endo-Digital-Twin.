import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import os
import sys

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from visualization import EndoVisualizer
from registry import load_predictor, check_model_availability
from explain import explain_prediction_change

# Page configuration
st.set_page_config(
    page_title="Endo Digital Twin",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    
    .main-header {
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
        color: #6B46C1;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    .prediction-card .metric {
        font-size: 3rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .explanation-box {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    
    .section-header {
        background: #f8fafc;
        border-left: 4px solid #6B46C1;
        padding: 0.75rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
DATA_FILE = "checkin_data.csv"

def load_data():
    """Load check-in data from CSV file"""
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame()

def save_data(df):
    """Save check-in data to CSV file"""
    df.to_csv(DATA_FILE, index=False)

def predict_grid(predict_fn, base_dict: Dict[str, float], x_name: str, y_name: str, 
                 xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    Build a 2D grid of predictions by varying X and Y features.
    
    Args:
        predict_fn: Function that takes a feature dict and returns a float
        base_dict: Base feature dictionary (other features held constant)
        x_name: Name of feature for X axis
        y_name: Name of feature for Y axis
        xs: Array of X values
        ys: Array of Y values
    
    Returns:
        2D numpy array of predictions (shape: len(ys), len(xs))
    """
    grid = np.zeros((len(ys), len(xs)))
    for i, y_val in enumerate(ys):
        for j, x_val in enumerate(xs):
            feats = base_dict.copy()
            feats[x_name] = x_val
            feats[y_name] = y_val
            grid[i, j] = predict_fn(feats)
    return grid

def top_contributors(baseline: Dict[str, float], scenario: Dict[str, float], 
                    coefficients: Optional[Dict[str, float]]) -> List[Tuple[str, float]]:
    """
    Calculate top contributors to pain change.
    
    Args:
        baseline: Baseline feature values
        scenario: Scenario feature values
        coefficients: Model coefficients (None if unavailable)
    
    Returns:
        List of (feature_name, impact) tuples sorted by absolute impact
    """
    contributions = {}
    key_features = ["sleep", "activity", "hydration", "mood", "stress"]
    
    for feat in key_features:
        if feat in baseline and feat in scenario:
            delta = scenario[feat] - baseline[feat]
            if coefficients and feat in coefficients:
                impact = delta * coefficients[feat]
            else:
                impact = 0.0  # No coefficient available
            contributions[feat] = impact
    
    # Sort by absolute impact, descending
    sorted_contribs = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    return sorted_contribs

def create_scenario_heatmap(predictor, base_dict: Dict[str, float], 
                            x_name: str, y_name: str, x_val: float, y_val: float,
                            baseline_x: float, baseline_y: float) -> go.Figure:
    """
    Create an interactive scenario heatmap.
    
    Args:
        predictor: Predictor object with predict() method
        base_dict: Base feature dictionary
        x_name: X axis feature name
        y_name: Y axis feature name
        x_val: Current scenario X value
        y_val: Current scenario Y value
        baseline_x: Baseline X value
        baseline_y: Baseline Y value
    
    Returns:
        Plotly figure
    """
    # Define ranges for each feature
    ranges = {
        'sleep': (3.0, 11.0, 25),
        'stress': (0.0, 10.0, 25),
        'activity': (0.0, 10.0, 25),
        'hydration': (0.0, 12.0, 25),
        'mood': (0.0, 10.0, 25)
    }
    
    x_min, x_max, x_steps = ranges[x_name]
    y_min, y_max, y_steps = ranges[y_name]
    
    xs = np.linspace(x_min, x_max, x_steps)
    ys = np.linspace(y_min, y_max, y_steps)
    
    # Build prediction grid
    grid = predict_grid(predictor.predict, base_dict, x_name, y_name, xs, ys)
    
    # Human-friendly labels
    labels = {
        'sleep': 'Sleep (hours)',
        'stress': 'Stress Level',
        'activity': 'Activity Level',
        'hydration': 'Hydration Level',
        'mood': 'Mood Level'
    }
    
    # Create heatmap with high-contrast colorscale (blue to red)
    # High-contrast custom colorscale: dark blue -> medium blue -> medium red -> dark red
    custom_colorscale = [
        [0.0, 'rgb(0, 0, 255)'],      # Bright blue (low pain)
        [0.33, 'rgb(100, 149, 237)'], # Cornflower blue
        [0.67, 'rgb(255, 69, 0)'],    # Red-orange
        [1.0, 'rgb(255, 0, 0)']       # Bright red (high pain)
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=grid,
        x=xs,
        y=ys,
        colorscale=custom_colorscale,
        zmin=0,
        zmax=10,
        colorbar=dict(title=dict(text="Predicted Pain", font=dict(color='#2D3748', size=12)))
    ))
    
    # Add scenario marker (X)
    fig.add_trace(go.Scatter(
        x=[x_val],
        y=[y_val],
        mode='markers+text',
        marker=dict(symbol='x', size=15, color='white', line=dict(width=2, color='black')),
        text=['You'],
        textposition='top center',
        name='Your Scenario',
        showlegend=False
    ))
    
    # Add baseline marker (dot)
    fig.add_trace(go.Scatter(
        x=[baseline_x],
        y=[baseline_y],
        mode='markers+text',
        marker=dict(symbol='circle', size=12, color='white', line=dict(width=2, color='black')),
        text=['Baseline'],
        textposition='top center',
        name='Baseline',
        showlegend=False
    ))
    
    fig.update_layout(
        title="Interactive Scenario Heatmap",
        xaxis_title=labels[x_name],
        yaxis_title=labels[y_name],
        template='plotly_white',
        height=420,
        title_font=dict(color='#2D3748', size=16),
        font=dict(color='#2D3748', size=12),
        xaxis=dict(title_font=dict(color='#2D3748', size=12)),
        yaxis=dict(title_font=dict(color='#2D3748', size=12))
    )
    
    return fig

def generate_doctor_summary(simulated_pain: float, baseline_pain: float, 
                           pain_change: float, pi_halfwidth: float,
                           baseline: Dict[str, float], scenario: Dict[str, float],
                           coefficients: Optional[Dict[str, float]]) -> Tuple[str, List[str]]:
    """
    Generate doctor-style summary and educational suggestions.
    
    Returns:
        Tuple of (summary_sentence, list_of_suggestions)
    """
    # Get top contributors
    contributors = top_contributors(baseline, scenario, coefficients)
    top_2_3 = contributors[:3]
    
    # Format contributors
    contrib_texts = []
    for name, impact in top_2_3:
        if abs(impact) > 0.01:  # Only show meaningful contributions
            name_display = name.capitalize()
            sign = '-' if impact < 0 else '+'
            contrib_texts.append(f"{name_display} ({sign}{abs(impact):.2f})")
    
    if contrib_texts:
        contrib_str = ", ".join(contrib_texts)
    else:
        contrib_str = "modest contributions from multiple factors"
    
    # Summary sentence
    summary = (
        f"Summary: Predicted pain is {simulated_pain:.1f}/10 "
        f"(change vs baseline: {pain_change:+.2f}, uncertainty ±{pi_halfwidth:.2f}). "
        f"Biggest modeled contributors: {contrib_str}."
    )
    
    # Educational suggestions based on thresholds
    suggestions = []
    
    if scenario.get('sleep', 8.0) < 7:
        suggestions.append("Aim for ~7–9 hours of sleep per night for better symptom management.")
    
    if scenario.get('stress', 5.0) >= 6:
        suggestions.append("Brief daily stress-reduction practices (e.g., 5–10 min breathing exercises or gentle movement) may help.")
    
    if scenario.get('hydration', 8.0) < 6:
        suggestions.append("Increase fluids gradually throughout the day (water, herbal teas) to support overall wellness.")
    
    if scenario.get('activity', 5.0) < 4:
        suggestions.append("Light, regular movement (walking, stretching) can help maintain flexibility and may ease discomfort.")
    
    if scenario.get('mood', 7.0) <= 4:
        suggestions.append("Consider brief mood-supporting activities (time outdoors, favorite music, connecting with others).")
    
    if not suggestions:
        suggestions.append("Inputs are near helpful ranges—keep routines steady and track patterns over time.")
    
    return summary, suggestions

def digital_twin_tab():
    """Digital Twin Simulator - Main tab"""
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; color: #64748b; font-size: 1.1rem;">
        Explore how different lifestyle factors might affect pain levels
    </div>
    """, unsafe_allow_html=True)
    
    # Check model availability
    model_availability = check_model_availability()
    available_models = [model for model, available in model_availability.items() if available]
    
    if not available_models:
        st.warning("⚠️ No trained models available. Please run the training scripts first:")
        st.code("""
        python src/synth.py
        python src/train_elasticnet.py
        python src/train_rf.py
        """)
        return
    
    # Use form for better UX
    with st.form("simulation_form", border=False):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Model Selection
            model_choice = st.selectbox("**Model Type**", available_models, 
                                      help="ElasticNet provides interpretable predictions. RandomForest captures non-linear patterns.")
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacer for alignment
        
        st.markdown('<div class="section-header">🏥 Input Your Information</div>', unsafe_allow_html=True)
        
        # Group parameters more logically
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Physical Health**")
            sleep_sim = st.slider("😴 Sleep Hours", 3.0, 11.0, 8.0, 0.5)
            activity_sim = st.slider("🏃 Activity Level", 0, 10, 5)
            hydration_sim = st.slider("💧 Hydration Level", 0, 12, 8)
        
        with col2:
            st.markdown("**Mental & Emotional**")
            stress_sim = st.slider("😰 Stress Level", 0, 10, 5)
            mood_sim = st.slider("😊 Mood Level", 0, 10, 7)
        
        with col3:
            st.markdown("**Symptoms & Cycle**")
            period_sim = st.selectbox("📅 Menstrual Phase", 
                                    [-2, -1, 0, 1, 2],
                                    format_func=lambda x: {
                                        -2: "Ovulation (-2)",
                                        -1: "Pre-menstrual (-1)",
                                        0: "Menstrual (0)",
                                        1: "Post-menstrual (+1)",
                                        2: "Follicular (+2)"
                                    }[x],
                                    index=2)
            gi_sim = st.checkbox("🤢 GI Symptoms")
            meds_sim = st.checkbox("💊 Taking NSAID")
        
        # Submit button
        submitted = st.form_submit_button("🔮 Generate Prediction", use_container_width=True, type="primary")
    
    # Show results only after form submission
    if submitted:
        try:
            # Load selected model
            predictor = load_predictor(model_choice)
            meta = predictor.get_metadata()
            
            # Baseline features
            baseline_features = {
                'sleep': 8.0,
                'stress': 5,
                'activity': 5,
                'period_phase': 0,
                'gi': 0,
                'meds': 0,
                'mood': 7,
                'hydration': 8
            }
            
            # Simulated features
            simulated_features = {
                'sleep': sleep_sim,
                'stress': stress_sim,
                'activity': activity_sim,
                'period_phase': period_sim,
                'gi': 1 if gi_sim else 0,
                'meds': 1 if meds_sim else 0,
                'mood': mood_sim,
                'hydration': hydration_sim
            }
            
            # Calculate predictions
            baseline_pain = predictor.predict(baseline_features)
            simulated_pain = predictor.predict(simulated_features)
            pain_change = simulated_pain - baseline_pain
            
            pi_halfwidth = meta.get('pi_halfwidth', 0.7)
            
            # Calculate individual feature effects using coefficients
            coefficients = meta.get('coefficients', {})
            feature_impacts = {}
            
            # Focus on the 5 key features requested
            key_features = ['sleep', 'activity', 'hydration', 'mood', 'stress']
            
            if coefficients:
                # ElasticNet: use coefficients directly
                for feature in key_features:
                    if feature in coefficients and feature in baseline_features and feature in simulated_features:
                        delta = simulated_features[feature] - baseline_features[feature]
                        impact = delta * coefficients[feature]
                        feature_impacts[feature] = impact
            else:
                # RandomForest: estimate using feature importances and deltas
                feature_importances = meta.get('feature_importances', {})
                if feature_importances:
                    # Normalize importances to get relative weights
                    total_importance = sum(abs(v) for v in feature_importances.values()) if feature_importances else 1
                    for feature in key_features:
                        if feature in feature_importances and feature in baseline_features and feature in simulated_features:
                            delta = simulated_features[feature] - baseline_features[feature]
                            # Estimate impact: delta * (importance / total) * pain_change
                            relative_importance = abs(feature_importances[feature]) / total_importance if total_importance > 0 else 0
                            impact = delta * relative_importance * (pain_change / max(abs(pain_change), 0.1))
                            feature_impacts[feature] = impact
            
            # Calculate individual effects for KPIs
            sleep_effect = feature_impacts.get('sleep', 0)
            stress_effect = feature_impacts.get('stress', 0)
            activity_effect = feature_impacts.get('activity', 0)
            hydration_effect = feature_impacts.get('hydration', 0)
            
            # Results Summary Dashboard
            st.markdown('<div class="section-header">📊 Results Summary</div>', unsafe_allow_html=True)
            
            # 1. KPI Metrics Row
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Predicted Pain", f"{simulated_pain:.1f}/10", 
                         delta=f"{pain_change:+.2f}", delta_color="inverse")
            
            with col2:
                st.metric("Sleep Effect", f"{sleep_effect:+.2f}", 
                         delta="Better" if sleep_effect < 0 else "Worse" if sleep_effect > 0 else "Neutral",
                         delta_color="inverse" if sleep_effect < 0 else "normal")
            
            with col3:
                st.metric("Stress Effect", f"{stress_effect:+.2f}", 
                         delta="Better" if stress_effect < 0 else "Worse" if stress_effect > 0 else "Neutral",
                         delta_color="inverse" if stress_effect < 0 else "normal")
            
            with col4:
                st.metric("Net Change", f"{pain_change:+.2f}", 
                         delta="vs Baseline", delta_color="inverse" if pain_change < 0 else "normal")
            
            with col5:
                st.metric("Activity Effect", f"{activity_effect:+.2f}",
                         delta="Better" if activity_effect < 0 else "Worse" if activity_effect > 0 else "Neutral",
                         delta_color="inverse" if activity_effect < 0 else "normal")
            
            st.divider()
            
            # 2 & 3. Feature Impact Bar Chart and Pain Profile Radar Chart side by side
            visualizer = EndoVisualizer()
            col1, col2 = st.columns(2)
            
            with col1:
                if feature_impacts:
                    fig_impact = visualizer.create_feature_impact_chart(feature_impacts)
                    st.plotly_chart(fig_impact, use_container_width=True)
            
            with col2:
                user_profile = {
                    'Sleep': sleep_sim,
                    'Activity': activity_sim,
                    'Hydration': hydration_sim,
                    'Mood': mood_sim,
                    'Low Stress': 10 - stress_sim  # Invert stress so higher = better
                }
                
                ideal_profile = {
                    'Sleep': 8.0,
                    'Activity': 8.0,
                    'Hydration': 10.0,
                    'Mood': 8.0,
                    'Low Stress': 8.0  # Low stress = 8 means stress = 2
                }
                
                fig_radar = visualizer.create_pain_profile_radar(user_profile, ideal_profile)
                st.plotly_chart(fig_radar, use_container_width=True)
            
            st.divider()
            
            # 4. What-If Explanation Sentence
            if coefficients or feature_impacts:
                # Generate plain English explanation
                explanations = []
                if abs(sleep_effect) > 0.1:
                    direction = "improved" if sleep_effect < 0 else "worsened"
                    explanations.append(f"{'Better' if sleep_effect < 0 else 'Reduced'} sleep {direction} pain by {abs(sleep_effect):.2f} points")
                
                if abs(hydration_effect) > 0.1:
                    direction = "improved" if hydration_effect < 0 else "worsened"
                    explanations.append(f"{'Better' if hydration_effect < 0 else 'Lower'} hydration {direction} pain by {abs(hydration_effect):.2f} points")
                
                if abs(stress_effect) > 0.1:
                    direction = "increased" if stress_effect > 0 else "decreased"
                    explanations.append(f"{'Higher' if stress_effect > 0 else 'Lower'} stress {direction} pain by {abs(stress_effect):.2f} points")
                
                if abs(activity_effect) > 0.1:
                    direction = "improved" if activity_effect < 0 else "worsened"
                    explanations.append(f"{'Better' if activity_effect < 0 else 'Lower'} activity {direction} pain by {abs(activity_effect):.2f} points")
                
                if explanations:
                    explanation_text = ". ".join(explanations) + f" — net change: {pain_change:+.2f} points."
                else:
                    explanation_text = f"Overall, your lifestyle factors result in a {pain_change:+.2f} point change in predicted pain."
                
                st.markdown(f"""
                <div class="explanation-box">
                    <p style="margin: 0; font-size: 1.1rem; color: #1a202c;"><strong style="color: #1a202c;">💡 What-If Summary:</strong> <span style="color: #1a202c;">{explanation_text}</span></p>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            # 5. Well-being Balance Gauge
            # Calculate overall balance score (0-100)
            # Weighted mix: sleep (20%), activity (15%), hydration (15%), mood (20%), low stress (30%)
            balance_score = (
                (sleep_sim / 11.0) * 20 +  # Sleep normalized to 0-11
                (activity_sim / 10.0) * 15 +
                (hydration_sim / 12.0) * 15 +
                (mood_sim / 10.0) * 20 +
                ((10 - stress_sim) / 10.0) * 30  # Low stress (inverted)
            )
            balance_score = max(0, min(100, balance_score))  # Clamp to 0-100
            
            fig_gauge = visualizer.create_wellbeing_gauge(balance_score)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.divider()
            
            # Part A: Interactive Scenario Heatmap
            st.markdown('<div class="section-header">🗺️ Interactive Scenario Heatmap</div>', unsafe_allow_html=True)
            
            # Feature selection dropdowns
            available_features = {
                'sleep': 'Sleep',
                'stress': 'Stress',
                'activity': 'Activity',
                'hydration': 'Hydration',
                'mood': 'Mood'
            }
            
            col1, col2 = st.columns(2)
            with col1:
                x_feature = st.selectbox(
                    "X Axis Feature",
                    options=list(available_features.keys()),
                    format_func=lambda x: available_features[x],
                    key="heatmap_x"
                )
            with col2:
                # Y axis options exclude the X axis selection
                y_options = [k for k in available_features.keys() if k != x_feature]
                # Default to first available option (index 0)
                y_feature = st.selectbox(
                    "Y Axis Feature",
                    options=y_options,
                    format_func=lambda x: available_features[x],
                    index=0,
                    key="heatmap_y"
                )
            
            # Create base dict for heatmap (hold other features constant at scenario values)
            # We'll vary x_feature and y_feature, so they don't need to be in base_dict
            # but predict_grid will copy and set them anyway, so including them is fine
            heatmap_base = simulated_features.copy()
            
            # Get current scenario and baseline values for X and Y
            scenario_x = float(simulated_features.get(x_feature, 0))
            scenario_y = float(simulated_features.get(y_feature, 0))
            baseline_x = float(baseline_features.get(x_feature, 0))
            baseline_y = float(baseline_features.get(y_feature, 0))
            
            # Create heatmap
            fig_heatmap = create_scenario_heatmap(
                predictor, heatmap_base, x_feature, y_feature,
                scenario_x, scenario_y, baseline_x, baseline_y
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            st.caption("💡 Tip: Switch axes to explore other pairs. Darker = higher predicted pain.")
            
            st.divider()
            
            # Part B: Doctor-Style Summary Block
            st.markdown('<div class="section-header" style="color: #1a202c;">👩‍⚕️ Doctor-Style Summary (Educational)</div>', unsafe_allow_html=True)
            
            # Try to load coefficients and pi_halfwidth from ElasticNet model
            try:
                en_path = "models/model_en.pkl"
                if os.path.exists(en_path):
                    en_obj = joblib.load(en_path)
                    en_meta = en_obj.get("meta", {})
                    en_coefficients = en_meta.get("coefficients", coefficients)
                    en_pi_hw = en_meta.get("pi_halfwidth", pi_halfwidth)
                else:
                    en_coefficients = coefficients
                    en_pi_hw = pi_halfwidth
            except Exception:
                en_coefficients = coefficients
                en_pi_hw = pi_halfwidth
            
            # Generate summary
            summary_text, suggestions = generate_doctor_summary(
                simulated_pain, baseline_pain, pain_change, en_pi_hw,
                baseline_features, simulated_features, en_coefficients
            )
            
            # Display summary sentence
            st.markdown(f"""
            <div style="background: #f8fafc; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #6B46C1; margin-bottom: 1rem;">
                <p style="margin: 0; color: #1a202c; font-size: 1rem;"><strong>{summary_text}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display educational suggestions
            if suggestions:
                st.info("**Educational suggestions (not medical advice):**\n\n" + "\n\n".join([f"• {s}" for s in suggestions]))
            
            # Sources caption
            st.caption("📚 **Sources:** General guidance informed by evidence-based resources from NIH, Mayo Clinic, ACOG, NICE, and WHO. This information is for educational purposes only and does not constitute medical advice. Always consult with a healthcare provider for personalized medical guidance.")
            
            # Optional: 7-day trendline if data exists
            df = load_data()
            if not df.empty and len(df) >= 2:
                df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
                df = df.sort_values('date')
                recent = df.tail(7)
                
                if len(recent) > 1:
                    avg_pain_recent = recent['pain_level'].mean()
                    avg_pain_prev = df.tail(14).head(7)['pain_level'].mean() if len(df) >= 7 else avg_pain_recent
                    pain_trend = avg_pain_recent - avg_pain_prev
                    
                    avg_sleep_recent = recent['sleep_hours'].mean()
                    avg_sleep_prev = df.tail(14).head(7)['sleep_hours'].mean() if len(df) >= 7 else avg_sleep_recent
                    sleep_trend = avg_sleep_recent - avg_sleep_prev
                    
                    st.divider()
                    st.markdown(f"""
                    <div style="background: #f0f9ff; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #0ea5e9;">
                        <p style="margin: 0; color: #0c4a6e;"><strong>📈 Past Week Trend:</strong> 
                        Pain {'↓' if pain_trend < 0 else '↑'} {abs(pain_trend):.1f} while sleep {'↑' if sleep_trend > 0 else '↓'} {abs(sleep_trend):.1f} hrs</p>
                    </div>
                    """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            return
    else:
        # Show placeholder before submission
        st.info("👆 Please fill in your information above and click 'Generate Prediction' to see results.")

def tracking_tab():
    """Data tracking and visualization"""
    st.markdown('<div class="section-header">📊 Your Data</div>', unsafe_allow_html=True)
    
    df = load_data()
    
    if df.empty:
        st.info("No tracking data yet. Use the simulator to explore predictions.")
        return
    
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
    df = df.dropna(subset=['date'])  # Remove rows with invalid dates
    df = df.sort_values('date')
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_pain = df['pain_level'].mean()
        st.metric("Average Pain", f"{avg_pain:.1f}/10")
    
    with col2:
        max_pain = df['pain_level'].max()
        st.metric("Worst Day", f"{max_pain:.1f}/10")
    
    with col3:
        total_entries = len(df)
        st.metric("Total Entries", total_entries)
    
    with col4:
        days_tracked = (df['date'].max() - df['date'].min()).days + 1
        st.metric("Days Tracked", days_tracked)
    
    # Charts
    visualizer = EndoVisualizer()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pain = visualizer.create_pain_timeline(df)
        st.plotly_chart(fig_pain, use_container_width=True)
    
    with col2:
        fig_dist = visualizer.create_pain_distribution(df)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Quick check-in form
    st.markdown('<div class="section-header">📝 Log Your Day</div>', unsafe_allow_html=True)
    
    with st.form("quick_checkin", border=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pain_level = st.slider("Pain Level (0-10)", 0, 10, 5)
            sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=8.0, step=0.5)
            activity_level = st.slider("Activity (0-10)", 0, 10, 5)
        
        with col2:
            mood = st.selectbox("Mood", ["Terrible", "Poor", "Okay", "Good", "Excellent"])
            stress_level = st.slider("Stress (0-10)", 0, 10, 5)
            period_phase = st.selectbox("Menstrual Phase", 
                                       ["-2 (Ovulation)", "-1 (Pre-menstrual)", 
                                        "0 (Menstrual)", "+1 (Post-menstrual)", "+2 (Follicular)"])
        
        with col3:
            gi_symptoms = st.selectbox("GI Symptoms", ["No", "Yes"])
            nsaid_taken = st.checkbox("Took NSAID")
            other_meds = st.text_input("Other Medications", placeholder="Optional")
        
        submitted = st.form_submit_button("💾 Save Entry", use_container_width=True)
        
        if submitted:
            new_entry = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'time': datetime.now().strftime('%H:%M'),
                'pain_level': pain_level,
                'mood': mood,
                'sleep_hours': sleep_hours,
                'stress_level': stress_level,
                'activity_level': activity_level,
                'period_phase': period_phase,
                'gi_symptoms': gi_symptoms,
                'nsaid_taken': nsaid_taken,
                'other_meds': other_meds
            }
            
            new_df = pd.DataFrame([new_entry])
            existing_df = df if not df.empty else pd.DataFrame()
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            save_data(updated_df)
            
            st.success("✅ Entry saved!")
    
    # Recent data
    if not df.empty:
        st.markdown('<div class="section-header">Recent Entries</div>', unsafe_allow_html=True)
        display_df = df[['date', 'pain_level', 'mood', 'sleep_hours', 'stress_level']].copy()
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
        st.dataframe(display_df.tail(10), use_container_width=True, hide_index=True)

def introduction_tab():
    """Introduction tab with app overview"""
    st.markdown('<div class="main-header">🩸 Endo Digital Twin</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1rem; color: #64748b; font-size: 1.1rem;">
        An educational tool for exploring endometriosis symptom patterns and lifestyle factors
    </div>
    """, unsafe_allow_html=True)
    
    # Display the image full width
    st.image("women_endo.png", use_container_width=True)
    
    # Scroll indicator
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <div style="color: #667eea; font-size: 2rem; animation: bounce 2s infinite;">⬇️</div>
        <p style="color: #9ca3af; font-size: 0.9rem; margin-top: 0.5rem;">Scroll to learn more</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <style>
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(10px); }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # What is this app section
    st.markdown("""
    <div style="background: #f8fafc; padding: 2rem; border-radius: 1rem; margin-bottom: 2rem; border-left: 4px solid #6B46C1;">
        <h2 style="color: #6B46C1; margin-top: 0;">📌 About This App</h2>
        <p style="font-size: 1.1rem; line-height: 1.8; color: #2D3748;">
            The <strong>Endo Digital Twin</strong> is an educational application designed to help individuals 
            understand how different lifestyle factors might influence endometriosis symptoms. This tool combines 
            machine learning predictions with intuitive visualization to explore "what-if" scenarios and track 
            personal patterns over time.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: #e0e7ff; padding: 1.5rem; border-radius: 1rem; margin-bottom: 1rem;">
            <h3 style="color: #4f46e5; margin-top: 0;">🧪 Simulator</h3>
            <p style="color: #1e1b4b;">
                Explore how different lifestyle factors might affect pain levels. Adjust sleep, stress, 
                activity, hydration, and menstrual cycle phase to see potential impacts on symptom severity.
                Choose between ElasticNet (interpretable) or RandomForest (non-linear patterns) models.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #dcfce7; padding: 1.5rem; border-radius: 1rem; margin-bottom: 1rem;">
            <h3 style="color: #15803d; margin-top: 0;">📊 Tracking</h3>
            <p style="color: #14532d;">
                Log your daily symptoms, mood, sleep, and activities. Visualize patterns over time with 
                interactive charts to identify trends and correlations in your data.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Important Disclaimer
    st.markdown("""
    <div style="background: #fef2f2; padding: 2rem; border-radius: 1rem; margin-bottom: 2rem; border-left: 4px solid #ef4444;">
        <h3 style="color: #dc2626; margin-top: 0;">⚠️ Important Disclaimer</h3>
        <p style="font-size: 1rem; line-height: 1.8; color: #991b1b;">
            This application is for <strong>educational purposes only</strong> and is not medical advice. 
            The predictions are based on simplified models trained on synthetic data and may not reflect your 
            individual experience. Always consult with healthcare professionals for medical decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # How it works
    st.markdown("""
    <div style="background: #f0f9ff; padding: 2rem; border-radius: 1rem; margin-bottom: 2rem; border-left: 4px solid #0ea5e9;">
        <h3 style="color: #0c4a6e; margin-top: 0;">🔬 How It Works</h3>
        <p style="font-size: 1rem; line-height: 1.8; color: #0c4a6e; margin-bottom: 1rem;">
            The app uses machine learning models (ElasticNet and RandomForest) trained on synthetic endometriosis 
            data to predict pain levels based on various lifestyle factors. The models consider:
        </p>
        <ul style="color: #0c4a6e; line-height: 2; font-size: 1rem;">
            <li>😴 Sleep patterns and duration</li>
            <li>😰 Stress levels</li>
            <li>🏃 Physical activity</li>
            <li>💧 Hydration</li>
            <li>🩸 Menstrual cycle phase</li>
            <li>🤢 Gastrointestinal symptoms</li>
            <li>💊 Medication use</li>
            <li>😊 Mood and emotional state</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # References
    st.markdown("""
    <div style="background: #faf5ff; padding: 2rem; border-radius: 1rem; margin-bottom: 2rem; border-left: 4px solid #8b5cf6;">
        <h3 style="color: #5b21b6; margin-top: 0;">📚 References & Further Reading</h3>
        <div style="color: #4c1d95; line-height: 2; font-size: 0.95rem;">
            <p style="margin: 0.5rem 0;"><strong>Endometriosis Overview:</strong></p>
            <ul style="margin: 0.5rem 0;">
                <li>Bulun, S. E., et al. (2019). Endometriosis. <em>Endocrine Reviews</em>, 40(4), 1048-1079.</li>
                <li>Zondervan, K. T., et al. (2020). Endometriosis. <em>Nature Reviews Disease Primers</em>, 6(1), 9.</li>
            </ul>
            
            <p style="margin: 0.5rem 0;"><strong>Lifestyle Factors & Symptoms:</strong></p>
            <ul style="margin: 0.5rem 0;">
                <li>Nodler, J. L., et al. (2020). Lifestyle and behavioral modifications for endometriosis. <em>Current Opinion in Obstetrics and Gynecology</em>, 32(4), 227-233.</li>
                <li>Agarwal, S. K., et al. (2019). Clinical diagnosis of endometriosis: a call to action. <em>American Journal of Obstetrics and Gynecology</em>, 220(4), 354.e1-354.e12.</li>
            </ul>
            
            <p style="margin: 0.5rem 0;"><strong>Digital Health Applications:</strong></p>
            <ul style="margin: 0.5rem 0;">
                <li>Nnoaham, K. E., et al. (2011). Impact of endometriosis on quality of life. <em>Human Reproduction</em>, 26(10), 2748-2756.</li>
                <li>World Endometriosis Society. (2021). World Endometriosis Society consensus on endometriosis.</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Getting Started
    st.markdown("""
    <div style="background: #fff7ed; padding: 1.5rem; border-radius: 1rem; text-align: center;">
        <h3 style="color: #c2410c; margin-top: 0;">🚀 Getting Started</h3>
        <p style="color: #9a3412; font-size: 1.1rem;">
            Navigate to the <strong>🧪 Simulator</strong> tab to explore predictions or the 
            <strong>📊 Tracking</strong> tab to log your data and view patterns.
        </p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main app function"""
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🏠 Introduction", "🧪 Simulator", "📊 Tracking"])
    
    with tab1:
        introduction_tab()
    
    with tab2:
        digital_twin_tab()
    
    with tab3:
        tracking_tab()

if __name__ == "__main__":
    main()
