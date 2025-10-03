import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import os
import sys

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model import EndoPredictionModel
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

# Custom CSS for enhanced theme
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Main header with gradient */
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
        text-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Enhanced tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: linear-gradient(90deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 0.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3.5rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
        border-radius: 0.75rem;
        color: #475569;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        border: 2px solid transparent;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: 2px solid #4f46e5;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* Enhanced metric cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
    
    /* Enhanced prediction cards */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        color: white;
        padding: 2rem;
        border-radius: 1.5rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, transparent 100%);
        pointer-events: none;
    }
    
    .prediction-card h3 {
        margin: 0 0 1rem 0;
        font-size: 1.2rem;
        font-weight: 600;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .prediction-card .metric {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Enhanced info boxes */
    .stInfo {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border: 1px solid #3b82f6;
        border-radius: 1rem;
        padding: 1rem;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border: 1px solid #22c55e;
        border-radius: 1rem;
        padding: 1rem;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 1px solid #f59e0b;
        border-radius: 1rem;
        padding: 1rem;
    }
    
    .stError {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border: 1px solid #ef4444;
        border-radius: 1rem;
        padding: 1rem;
    }
    
    /* Enhanced sliders */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Enhanced selectboxes */
    .stSelectbox > div > div {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 2px solid #e2e8f0;
        border-radius: 0.75rem;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #667eea;
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.1);
    }
    
    /* Enhanced checkboxes */
    .stCheckbox > label > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Enhanced buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.75rem;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Enhanced dataframes */
    .stDataFrame {
        border-radius: 1rem;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* Custom section headers */
    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 1rem;
        margin: 1.5rem 0 1rem 0;
        font-weight: 600;
        font-size: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.2);
    }
    
    /* Model info cards */
    .model-info-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border: 2px solid #cbd5e1;
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .model-info-card:hover {
        border-color: #667eea;
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
    
    /* Explanation box */
    .explanation-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 2px solid #f59e0b;
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(245, 158, 11, 0.2);
    }
    
    /* Feature importance bars */
    .feature-bar {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 8px;
        border-radius: 4px;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .feature-bar:hover {
        height: 12px;
        box-shadow: 0 2px 4px rgba(102, 126, 234, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'checkin_data' not in st.session_state:
    st.session_state.checkin_data = []

# Data storage file
DATA_FILE = "checkin_data.csv"

def load_data():
    """Load check-in data from CSV file"""
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame()

def save_data(df):
    """Save check-in data to CSV file"""
    df.to_csv(DATA_FILE, index=False)

def checkin_tab():
    """Check-in tab for daily logging"""
    st.markdown('<div class="section-header">📋 Daily Check-in</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; color: #64748b; font-size: 1.1rem;">
        🌟 Log your daily symptoms and activities to track patterns over time
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("checkin_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
                        padding: 1.5rem; border-radius: 1rem; margin-bottom: 1.5rem; 
                        border: 2px solid #f59e0b;">
                <h3 style="color: #92400e; margin-top: 0; text-align: center;">🌙 Physical Symptoms</h3>
            </div>
            """, unsafe_allow_html=True)
            
            pain_level = st.slider("😰 Pain Level (0-10)", 0, 10, 5, help="Rate your pain from 0 (no pain) to 10 (severe pain)")
            sleep_hours = st.number_input("😴 Sleep Hours", min_value=0.0, max_value=24.0, value=8.0, step=0.5)
            activity_level = st.slider("🏃 Activity Level (0-10)", 0, 10, 5, help="Rate your activity level from 0 (very low) to 10 (very high)")
            gi_symptoms = st.selectbox("🤢 GI Symptoms", ["No", "Yes"], help="Gastrointestinal symptoms like bloating, nausea, etc.")
            
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); 
                        padding: 1.5rem; border-radius: 1rem; margin-bottom: 1.5rem; 
                        border: 2px solid #3b82f6;">
                <h3 style="color: #1e40af; margin-top: 0; text-align: center;">🧠 Mental & Emotional</h3>
            </div>
            """, unsafe_allow_html=True)
            
            mood = st.selectbox("😊 Mood", ["Excellent", "Good", "Okay", "Poor", "Terrible"])
            stress_level = st.slider("😰 Stress Level (0-10)", 0, 10, 5, help="Rate your stress from 0 (no stress) to 10 (very stressed)")
            period_phase = st.selectbox("🩸 Period Phase", 
                                     ["-2 (Ovulation)", "-1 (Pre-menstrual)", "0 (Menstrual)", "+1 (Post-menstrual)", "+2 (Follicular)"],
                                     help="Where you are in your menstrual cycle")
            
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); 
                    padding: 1.5rem; border-radius: 1rem; margin-bottom: 1.5rem; 
                    border: 2px solid #8b5cf6;">
            <h3 style="color: #6b21a8; margin-top: 0; text-align: center;">💊 Medication</h3>
        </div>
        """, unsafe_allow_html=True)
        
        nsaid_taken = st.checkbox("💊 Took NSAID (Ibuprofen, Naproxen, etc.)")
        other_meds = st.text_input("💉 Other Medications (optional)", placeholder="e.g., Birth control, supplements")
        
        submitted = st.form_submit_button("💾 Save Check-in", use_container_width=True)
        
        if submitted:
            # Create new entry
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
            
            # Load existing data and append new entry
            df = load_data()
            new_df = pd.DataFrame([new_entry])
            df = pd.concat([df, new_df], ignore_index=True)
            save_data(df)
            
            st.success("✅ Check-in saved successfully!")
            st.balloons()

def digital_twin_tab():
    """Digital Twin Simulator tab"""
    st.markdown('<div class="section-header">🧪 Digital Twin Simulator</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; color: #64748b; font-size: 1.1rem;">
        🌟 Explore 'what-if' scenarios to understand how different factors might affect your pain levels
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
    
    # Initialize visualizer
    visualizer = EndoVisualizer()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Model Selection with enhanced styling
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); 
                    padding: 1.5rem; border-radius: 1rem; margin-bottom: 1.5rem; 
                    border: 2px solid #cbd5e1;">
            <h3 style="color: #475569; margin-top: 0; text-align: center;">🤖 Model Selection</h3>
        </div>
        """, unsafe_allow_html=True)
        
        model_choice = st.selectbox("Choose Model", available_models, 
                                  help="ElasticNet provides interpretable linear predictions. RandomForest captures non-linear patterns.")
        
        # Parameters section with colorful icons
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
                    padding: 1.5rem; border-radius: 1rem; margin-bottom: 1.5rem; 
                    border: 2px solid #f59e0b;">
            <h3 style="color: #92400e; margin-top: 0; text-align: center;">⚙️ Adjust Parameters</h3>
            <p style="color: #92400e; text-align: center; margin-bottom: 0;">Move the sliders to simulate different conditions</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Physical parameters
        st.markdown("**🌙 Physical Health**")
        sleep_sim = st.slider("😴 Sleep Hours", 3.0, 11.0, 8.0, 0.5, key="sim_sleep")
        activity_sim = st.slider("🏃 Activity Level", 0, 10, 5, key="sim_activity")
        hydration_sim = st.slider("💧 Hydration Level", 0, 12, 8, key="sim_hydration")
        
        st.markdown("**🧠 Mental & Emotional**")
        stress_sim = st.slider("😰 Stress Level", 0, 10, 5, key="sim_stress")
        mood_sim = st.slider("😊 Mood Level", 0, 10, 7, key="sim_mood")
        
        st.markdown("**🩸 Menstrual Cycle**")
        period_sim = st.selectbox("📅 Period Phase", 
                                [-2, -1, 0, 1, 2],
                                format_func=lambda x: f"{x} ({'Ovulation' if x==-2 else 'Pre-menstrual' if x==-1 else 'Menstrual' if x==0 else 'Post-menstrual' if x==1 else 'Follicular'})",
                                index=2, key="sim_period")
        
        st.markdown("**💊 Symptoms & Medication**")
        gi_sim = st.checkbox("🤢 GI Symptoms", key="sim_gi")
        meds_sim = st.checkbox("💊 Taking NSAID", key="sim_meds")
        
    with col2:
        # Prediction section with enhanced styling
        st.markdown("""
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); 
                    padding: 1.5rem; border-radius: 1rem; margin-bottom: 1.5rem; 
                    border: 2px solid #3b82f6;">
            <h3 style="color: #1e40af; margin-top: 0; text-align: center;">🎯 Pain Prediction</h3>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            # Load selected model
            predictor = load_predictor(model_choice)
            meta = predictor.get_metadata()
            
            # Prepare features for prediction (convert to model format)
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
            
            # Calculate predictions using the ML model
            baseline_pain = predictor.predict(baseline_features)
            simulated_pain = predictor.predict(simulated_features)
            pain_change = simulated_pain - baseline_pain
            
            # Get prediction interval half-width
            pi_halfwidth = meta.get('pi_halfwidth', 0.7)
            
            # Display prediction cards with enhanced styling
            st.markdown(f"""
            <div class="prediction-card">
                <h3>📊 Baseline Pain</h3>
                <div class="metric">{baseline_pain:.1f}/10</div>
                <p style="margin: 0; opacity: 0.9;">Typical scenario</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Color-coded change indicator
            change_color = "#22c55e" if pain_change < 0 else "#ef4444" if pain_change > 0.5 else "#f59e0b"
            change_icon = "📉" if pain_change < 0 else "📈" if pain_change > 0.5 else "➡️"
            
            st.markdown(f"""
            <div class="prediction-card" style="background: linear-gradient(135deg, {change_color} 0%, #764ba2 50%, #f093fb 100%);">
                <h3>{change_icon} Simulated Pain</h3>
                <div class="metric">{simulated_pain:.1f}/10</div>
                <p style="margin: 0; opacity: 0.9;">Change: {pain_change:+.2f} points</p>
                <p style="margin: 0; opacity: 0.8; font-size: 0.9rem;">Uncertainty (±): {pi_halfwidth:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Model explanation for ElasticNet
            if model_choice == "ElasticNet" and 'coefficients' in meta:
                try:
                    explanation = explain_prediction_change(
                        baseline_features, simulated_features, 
                        meta['coefficients'], pain_change
                    )
                    st.markdown(f"""
                    <div class="explanation-box">
                        <h4 style="color: #92400e; margin-top: 0;">💡 AI Explanation</h4>
                        <p style="color: #92400e; margin-bottom: 0;">{explanation}</p>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.debug(f"Explanation error: {e}")
            
            # Create comparison chart
            fig = visualizer.create_comparison_chart(baseline_pain, simulated_pain)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            return
    
    # Model information section with enhanced styling
    if 'predictor' in locals():
        st.markdown("""
        <div class="section-header">📊 Model Information</div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="model-info-card">
                <h4 style="color: #475569; margin-top: 0;">🤖 Model Type</h4>
                <p style="font-size: 1.2rem; font-weight: 600; color: #667eea; margin: 0;">{}</p>
            </div>
            """.format(meta.get('type', 'unknown').upper()), unsafe_allow_html=True)
            
            st.markdown("""
            <div class="model-info-card">
                <h4 style="color: #475569; margin-top: 0;">📈 R² Score</h4>
                <p style="font-size: 1.2rem; font-weight: 600; color: #22c55e; margin: 0;">{:.3f}</p>
            </div>
            """.format(meta.get('r2', 0)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="model-info-card">
                <h4 style="color: #475569; margin-top: 0;">📏 MAE</h4>
                <p style="font-size: 1.2rem; font-weight: 600; color: #f59e0b; margin: 0;">{:.3f}</p>
            </div>
            """.format(meta.get('mae', 0)), unsafe_allow_html=True)
            
            st.markdown("""
            <div class="model-info-card">
                <h4 style="color: #475569; margin-top: 0;">🎯 Uncertainty</h4>
                <p style="font-size: 1.2rem; font-weight: 600; color: #8b5cf6; margin: 0;">±{:.3f}</p>
            </div>
            """.format(pi_halfwidth), unsafe_allow_html=True)
        
        with col3:
            if model_choice == "ElasticNet" and 'coefficients' in meta:
                st.markdown("""
                <div class="model-info-card">
                    <h4 style="color: #475569; margin-top: 0;">🔍 Top Coefficients</h4>
                """, unsafe_allow_html=True)
                sorted_coefs = sorted(meta['coefficients'].items(), 
                                    key=lambda x: abs(x[1]), reverse=True)[:3]
                for feature, coef in sorted_coefs:
                    direction = "↗️" if coef > 0 else "↘️"
                    color = "#ef4444" if coef > 0 else "#22c55e"
                    st.markdown(f"""
                    <p style="margin: 0.5rem 0; color: {color};">
                        {direction} <strong>{feature}</strong>: {coef:.3f}
                    </p>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            elif model_choice == "RandomForest" and 'feature_importances' in meta:
                st.markdown("""
                <div class="model-info-card">
                    <h4 style="color: #475569; margin-top: 0;">🌳 Top Features</h4>
                """, unsafe_allow_html=True)
                sorted_imps = sorted(meta['feature_importances'].items(), 
                                   key=lambda x: x[1], reverse=True)[:3]
                for feature, importance in sorted_imps:
                    st.markdown(f"""
                    <p style="margin: 0.5rem 0; color: #667eea;">
                        🔍 <strong>{feature}</strong>: {importance:.3f}
                    </p>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

def dashboard_tab():
    """Dashboard tab for data visualization"""
    st.markdown('<div class="section-header">📊 Dashboard</div>', unsafe_allow_html=True)
    
    # Initialize visualizer
    visualizer = EndoVisualizer()
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.info("No check-in data available yet. Please use the Check-in tab to log your first entry.")
        return
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Summary statistics with enhanced styling
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); 
                padding: 1.5rem; border-radius: 1rem; margin-bottom: 2rem; 
                border: 2px solid #cbd5e1;">
        <h3 style="color: #475569; margin-top: 0; text-align: center;">📈 Your Health Summary</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_pain = df['pain_level'].mean()
        st.markdown(f"""
        <div class="model-info-card">
            <h4 style="color: #475569; margin-top: 0;">😰 Average Pain</h4>
            <p style="font-size: 1.5rem; font-weight: 700; color: #ef4444; margin: 0;">{avg_pain:.1f}/10</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        max_pain = df['pain_level'].max()
        st.markdown(f"""
        <div class="model-info-card">
            <h4 style="color: #475569; margin-top: 0;">🔥 Worst Day</h4>
            <p style="font-size: 1.5rem; font-weight: 700; color: #dc2626; margin: 0;">{max_pain:.1f}/10</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_entries = len(df)
        st.markdown(f"""
        <div class="model-info-card">
            <h4 style="color: #475569; margin-top: 0;">📝 Total Entries</h4>
            <p style="font-size: 1.5rem; font-weight: 700; color: #22c55e; margin: 0;">{total_entries}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        days_tracked = (df['date'].max() - df['date'].min()).days + 1
        st.markdown(f"""
        <div class="model-info-card">
            <h4 style="color: #475569; margin-top: 0;">📅 Days Tracked</h4>
            <p style="font-size: 1.5rem; font-weight: 700; color: #3b82f6; margin: 0;">{days_tracked}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Pain over time using visualizer
        fig_pain = visualizer.create_pain_timeline(df)
        st.plotly_chart(fig_pain, use_container_width=True)
    
    with col2:
        # Pain distribution using visualizer
        fig_dist = visualizer.create_pain_distribution(df)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Additional charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Mood over time
        fig_mood = visualizer.create_mood_timeline(df)
        st.plotly_chart(fig_mood, use_container_width=True)
    
    with col2:
        # Correlation heatmap
        if len(df) > 1:  # Need at least 2 data points for correlation
            fig_corr = visualizer.create_correlation_heatmap(df)
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Need more data points to show correlation heatmap")
    
    # Data table
    st.markdown("### Recent Check-ins")
    display_df = df[['date', 'pain_level', 'mood', 'sleep_hours', 'stress_level', 'activity_level', 'period_phase']].copy()
    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
    st.dataframe(display_df.tail(10), use_container_width=True)

def about_tab():
    """About tab with app information"""
    st.markdown('<div class="section-header">ℹ️ About Endo Digital Twin</div>', unsafe_allow_html=True)
    
    # Purpose section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); 
                padding: 2rem; border-radius: 1rem; margin-bottom: 2rem; 
                border: 2px solid #3b82f6;">
        <h3 style="color: #1e40af; margin-top: 0; text-align: center;">🎯 Purpose</h3>
        <p style="color: #1e40af; font-size: 1.1rem; line-height: 1.6;">
            This is an <strong>educational demo</strong> designed to help people understand how different lifestyle factors 
            might influence endometriosis symptoms. It's not a medical device and should not be used for 
            medical diagnosis or treatment decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Digital Twin Concept
    st.markdown("""
    <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
                padding: 2rem; border-radius: 1rem; margin-bottom: 2rem; 
                border: 2px solid #f59e0b;">
        <h3 style="color: #92400e; margin-top: 0; text-align: center;">🧪 Digital Twin Concept</h3>
        <p style="color: #92400e; font-size: 1.1rem; line-height: 1.6; margin-bottom: 1rem;">
            The Digital Twin Simulator allows you to explore "what-if" scenarios by adjusting various factors:
        </p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
            <div style="background: rgba(255,255,255,0.3); padding: 1rem; border-radius: 0.5rem;">
                <strong>😴 Sleep patterns</strong> - How different sleep durations might affect pain
            </div>
            <div style="background: rgba(255,255,255,0.3); padding: 1rem; border-radius: 0.5rem;">
                <strong>😰 Stress levels</strong> - The relationship between stress and symptom severity
            </div>
            <div style="background: rgba(255,255,255,0.3); padding: 1rem; border-radius: 0.5rem;">
                <strong>🏃 Activity levels</strong> - Finding the right balance of physical activity
            </div>
            <div style="background: rgba(255,255,255,0.3); padding: 1rem; border-radius: 0.5rem;">
                <strong>💧 Hydration</strong> - Understanding the impact of proper hydration
            </div>
            <div style="background: rgba(255,255,255,0.3); padding: 1rem; border-radius: 0.5rem;">
                <strong>🩸 Menstrual cycle</strong> - How different cycle phases affect symptoms
            </div>
            <div style="background: rgba(255,255,255,0.3); padding: 1rem; border-radius: 0.5rem;">
                <strong>💊 Medications</strong> - The potential impact of NSAID use
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Data Tracking
    st.markdown("""
    <div style="background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); 
                padding: 2rem; border-radius: 1rem; margin-bottom: 2rem; 
                border: 2px solid #22c55e;">
        <h3 style="color: #166534; margin-top: 0; text-align: center;">📊 Data Tracking</h3>
        <p style="color: #166534; font-size: 1.1rem; line-height: 1.6; margin-bottom: 1rem;">
            The Check-in feature helps you:
        </p>
        <ul style="color: #166534; font-size: 1.1rem; line-height: 1.8;">
            <li>📈 Track daily symptoms and activities</li>
            <li>🔍 Identify patterns over time</li>
            <li>💡 Make more informed lifestyle choices</li>
            <li>👩‍⚕️ Prepare better questions for your healthcare provider</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); 
                padding: 2rem; border-radius: 1rem; margin-bottom: 2rem; 
                border: 2px solid #ef4444;">
        <h3 style="color: #dc2626; margin-top: 0; text-align: center;">⚠️ Important Disclaimer</h3>
        <ul style="color: #dc2626; font-size: 1.1rem; line-height: 1.8;">
            <li>This app is for <strong>educational purposes only</strong></li>
            <li>It is <strong>not medical advice</strong></li>
            <li>Always consult with healthcare professionals for medical decisions</li>
            <li>The predictions are simplified models and may not reflect your individual experience</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Future Development
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); 
                padding: 2rem; border-radius: 1rem; margin-bottom: 2rem; 
                border: 2px solid #8b5cf6;">
        <h3 style="color: #6b21a8; margin-top: 0; text-align: center;">🔬 Future Development</h3>
        <p style="color: #6b21a8; font-size: 1.1rem; line-height: 1.6;">
            This demo includes trained machine learning models (ElasticNet and RandomForest) 
            for sophisticated predictions and pattern recognition. The models are trained on 
            synthetic data and provide uncertainty estimates for better decision-making.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Support
    st.markdown("""
    <div style="background: linear-gradient(135deg, #fce7f3 0%, #fbcfe8 100%); 
                padding: 2rem; border-radius: 1rem; margin-bottom: 2rem; 
                border: 2px solid #ec4899;">
        <h3 style="color: #be185d; margin-top: 0; text-align: center;">💜 Support</h3>
        <p style="color: #be185d; font-size: 1.1rem; line-height: 1.6; text-align: center;">
            If you find this tool helpful, consider sharing it with others who might benefit from 
            understanding endometriosis patterns and management strategies.
        </p>
        <p style="color: #be185d; font-size: 1.2rem; font-weight: 600; text-align: center; margin-bottom: 0;">
            Built with ❤️ for the endometriosis community
        </p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main app function"""
    # Header
    st.markdown('<div class="main-header">🩸 Endo Digital Twin</div>', unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Check-in", "🧪 Digital Twin", "📊 Dashboard", "ℹ️ About"])
    
    with tab1:
        checkin_tab()
    
    with tab2:
        digital_twin_tab()
    
    with tab3:
        dashboard_tab()
    
    with tab4:
        about_tab()

if __name__ == "__main__":
    main()
