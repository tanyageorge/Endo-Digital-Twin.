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

# Page configuration
st.set_page_config(
    page_title="Endo Digital Twin",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for theme
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #6B46C1;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding-left: 1rem;
        padding-right: 1rem;
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        color: #6B46C1;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6B46C1;
        color: white;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #6B46C1;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #6B46C1, #14B8A6);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
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
    st.markdown('<h2 style="color: #6B46C1;">📋 Daily Check-in</h2>', unsafe_allow_html=True)
    st.markdown("Log your daily symptoms and activities to track patterns over time.")
    
    with st.form("checkin_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Physical Symptoms")
            pain_level = st.slider("Pain Level (0-10)", 0, 10, 5, help="Rate your pain from 0 (no pain) to 10 (severe pain)")
            sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=8.0, step=0.5)
            activity_level = st.slider("Activity Level (0-10)", 0, 10, 5, help="Rate your activity level from 0 (very low) to 10 (very high)")
            gi_symptoms = st.selectbox("GI Symptoms", ["No", "Yes"], help="Gastrointestinal symptoms like bloating, nausea, etc.")
            
        with col2:
            st.markdown("### Mental & Emotional")
            mood = st.selectbox("Mood", ["Excellent", "Good", "Okay", "Poor", "Terrible"])
            stress_level = st.slider("Stress Level (0-10)", 0, 10, 5, help="Rate your stress from 0 (no stress) to 10 (very stressed)")
            period_phase = st.selectbox("Period Phase", 
                                     ["-2 (Ovulation)", "-1 (Pre-menstrual)", "0 (Menstrual)", "+1 (Post-menstrual)", "+2 (Follicular)"],
                                     help="Where you are in your menstrual cycle")
            
        st.markdown("### Medication")
        nsaid_taken = st.checkbox("Took NSAID (Ibuprofen, Naproxen, etc.)")
        other_meds = st.text_input("Other Medications (optional)", placeholder="e.g., Birth control, supplements")
        
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
    st.markdown('<h2 style="color: #6B46C1;">🧪 Digital Twin Simulator</h2>', unsafe_allow_html=True)
    st.markdown("Explore 'what-if' scenarios to understand how different factors might affect your pain levels.")
    
    # Initialize model and visualizer
    model = EndoPredictionModel()
    visualizer = EndoVisualizer()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Adjust Parameters")
        st.markdown("Move the sliders to simulate different conditions:")
        
        sleep_sim = st.slider("Sleep Hours", 0.0, 12.0, 8.0, 0.5, key="sim_sleep")
        stress_sim = st.slider("Stress Level", 0, 10, 5, key="sim_stress")
        activity_sim = st.slider("Activity Level", 0, 10, 5, key="sim_activity")
        hydration_sim = st.slider("Hydration Level", 0, 10, 7, key="sim_hydration")
        
        period_sim = st.selectbox("Period Phase", 
                                ["-2 (Ovulation)", "-1 (Pre-menstrual)", "0 (Menstrual)", "+1 (Post-menstrual)", "+2 (Follicular)"],
                                key="sim_period")
        
        meds_sim = st.checkbox("Taking NSAID", key="sim_meds")
        gi_sim = st.selectbox("GI Symptoms", ["No", "Yes"], key="sim_gi")
        
    with col2:
        st.markdown("### Pain Prediction")
        
        # Prepare features for prediction
        baseline_features = {
            'sleep_hours': 8.0,
            'stress_level': 5,
            'activity_level': 5,
            'hydration_level': 7,
            'period_phase': '0 (Menstrual)',
            'nsaid_taken': False,
            'gi_symptoms': 'No'
        }
        
        simulated_features = {
            'sleep_hours': sleep_sim,
            'stress_level': stress_sim,
            'activity_level': activity_sim,
            'hydration_level': hydration_sim,
            'period_phase': period_sim,
            'nsaid_taken': meds_sim,
            'gi_symptoms': gi_sim
        }
        
        # Calculate predictions using the model
        baseline_pain = model.predict_pain(baseline_features)
        simulated_pain = model.predict_pain(simulated_features)
        
        # Display prediction cards
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.metric("Baseline Pain", f"{baseline_pain:.1f}/10", "Typical scenario")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        change = simulated_pain - baseline_pain
        st.metric("Simulated Pain", f"{simulated_pain:.1f}/10", f"{change:+.1f} from baseline")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Create comparison chart using visualizer
        fig = visualizer.create_comparison_chart(baseline_pain, simulated_pain)
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance section
    st.markdown("### Feature Impact")
    importance = model.get_feature_importance()
    
    # Create feature importance chart
    features = list(importance.keys())
    values = list(importance.values())
    
    fig_importance = px.bar(
        x=values,
        y=features,
        orientation='h',
        title="Feature Importance in Pain Prediction",
        labels={'x': 'Importance Score', 'y': 'Features'},
        color_discrete_sequence=[visualizer.color_primary]
    )
    
    fig_importance.update_layout(height=300)
    st.plotly_chart(fig_importance, use_container_width=True)

def dashboard_tab():
    """Dashboard tab for data visualization"""
    st.markdown('<h2 style="color: #6B46C1;">📊 Dashboard</h2>', unsafe_allow_html=True)
    
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
    
    # Summary statistics
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
    st.markdown('<h2 style="color: #6B46C1;">ℹ️ About Endo Digital Twin</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Purpose
    This is an **educational demo** designed to help people understand how different lifestyle factors 
    might influence endometriosis symptoms. It's not a medical device and should not be used for 
    medical diagnosis or treatment decisions.
    
    ### 🧪 Digital Twin Concept
    The Digital Twin Simulator allows you to explore "what-if" scenarios by adjusting various factors:
    - **Sleep patterns** - How different sleep durations might affect pain
    - **Stress levels** - The relationship between stress and symptom severity  
    - **Activity levels** - Finding the right balance of physical activity
    - **Hydration** - Understanding the impact of proper hydration
    - **Menstrual cycle** - How different cycle phases affect symptoms
    - **Medications** - The potential impact of NSAID use
    
    ### 📊 Data Tracking
    The Check-in feature helps you:
    - Track daily symptoms and activities
    - Identify patterns over time
    - Make more informed lifestyle choices
    - Prepare better questions for your healthcare provider
    
    ### ⚠️ Important Disclaimer
    - This app is for **educational purposes only**
    - It is **not medical advice**
    - Always consult with healthcare professionals for medical decisions
    - The predictions are simplified models and may not reflect your individual experience
    
    ### 🔬 Future Development
    This demo will eventually connect to trained machine learning models in the `src/` directory 
    for more sophisticated predictions and pattern recognition.
    
    ### 💜 Support
    If you find this tool helpful, consider sharing it with others who might benefit from 
    understanding endometriosis patterns and management strategies.
    """)
    
    st.markdown("---")
    st.markdown("**Built with ❤️ for the endometriosis community**")

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
