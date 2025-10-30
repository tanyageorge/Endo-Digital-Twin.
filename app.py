import streamlit as st
import pandas as pd
from datetime import datetime
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
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Model Selection
        model_choice = st.selectbox("**Model**", available_models, 
                                  help="ElasticNet provides interpretable predictions. RandomForest captures non-linear patterns.")
        
        st.markdown('<div class="section-header">Adjust Factors</div>', unsafe_allow_html=True)
        
        # Group parameters more logically
        sleep_sim = st.slider("😴 Sleep Hours", 3.0, 11.0, 8.0, 0.5)
        activity_sim = st.slider("🏃 Activity Level", 0, 10, 5)
        hydration_sim = st.slider("💧 Hydration Level", 0, 12, 8)
        
        stress_sim = st.slider("😰 Stress Level", 0, 10, 5)
        mood_sim = st.slider("😊 Mood Level", 0, 10, 7)
        
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
    
    with col2:
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
            
            # Display predictions
            st.markdown(f"""
            <div class="prediction-card">
                <h3>Baseline Pain</h3>
                <div class="metric">{baseline_pain:.1f}/10</div>
                <p style="margin: 0; opacity: 0.9;">Typical scenario</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Color-coded change
            change_color = "#22c55e" if pain_change < -0.1 else "#ef4444" if pain_change > 0.5 else "#f59e0b"
            
            st.markdown(f"""
            <div class="prediction-card" style="background: linear-gradient(135deg, {change_color} 0%, #764ba2 100%);">
                <h3>Simulated Pain</h3>
                <div class="metric">{simulated_pain:.1f}/10</div>
                <p style="margin: 0; opacity: 0.9;">Change: {pain_change:+.2f} points</p>
                <p style="margin: 0; opacity: 0.8; font-size: 0.9rem;">Uncertainty: ±{pi_halfwidth:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # AI Explanation
            if model_choice == "ElasticNet" and 'coefficients' in meta:
                try:
                    explanation = explain_prediction_change(
                        baseline_features, simulated_features, 
                        meta['coefficients'], pain_change
                    )
                    st.markdown(f"""
                    <div class="explanation-box">
                        <p style="margin: 0;"><strong>💡 Explanation:</strong> {explanation}</p>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    pass
            
            # Visualization
            visualizer = EndoVisualizer()
            fig = visualizer.create_comparison_chart(baseline_pain, simulated_pain)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            return

def tracking_tab():
    """Data tracking and visualization"""
    st.markdown('<div class="section-header">📊 Your Data</div>', unsafe_allow_html=True)
    
    df = load_data()
    
    if df.empty:
        st.info("No tracking data yet. Use the simulator to explore predictions.")
        return
    
    df['date'] = pd.to_datetime(df['date'])
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
