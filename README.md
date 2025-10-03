# 🩸 Endo Digital Twin

An educational Streamlit app for exploring endometriosis symptom patterns and "what-if" scenarios with machine learning predictions.

## 🎯 Features

- **📋 Daily Check-in**: Log pain levels, mood, sleep, stress, activity, period phase, GI symptoms, and medication use
- **🧪 Digital Twin Simulator**: Explore how different lifestyle factors might affect pain levels using trained ML models
- **📊 Dashboard**: Visualize your data with charts and summary statistics
- **ℹ️ About**: Learn about the app and its educational purpose

## 🤖 Machine Learning Models

The app includes two trained models for pain prediction:

- **ElasticNet**: Linear model with interpretable coefficients and natural language explanations
- **RandomForest**: Non-linear ensemble model for complex pattern recognition

Both models are trained on synthetic endometriosis data and provide uncertainty estimates.

## 🚀 Quick Start

### Option 1: Using Launch Scripts

**macOS/Linux:**
```bash
./run_app.sh
```

**Windows:**
```cmd
run_app.bat
```

### Option 2: Manual Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Generate synthetic data:
   ```bash
   python src/synth.py
   ```

3. Train the ML models:
   ```bash
   python src/train_elasticnet.py
   python src/train_rf.py
   ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```

5. Open your browser to `http://localhost:8501`

## 📁 Project Structure

```
Endo-Digital-Twin./
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── run_app.sh/.bat       # Launch scripts
├── .gitignore            # Git ignore file
├── data/                 # Generated data
│   └── synthetic.csv     # Synthetic training data
├── models/               # Trained models
│   ├── model_en.pkl      # ElasticNet model
│   └── model_rf.pkl      # RandomForest model
└── src/                  # Source code
    ├── __init__.py
    ├── synth.py          # Synthetic data generator
    ├── train_elasticnet.py # ElasticNet training
    ├── train_rf.py       # RandomForest training
    ├── registry.py       # Model registry/loader
    ├── explain.py        # Explanation utilities
    ├── model.py          # Mock prediction model
    └── visualization.py  # Chart utilities
```

## 🔬 ML Pipeline Details

### Data Generation (`src/synth.py`)
- Generates 8,000 synthetic samples with realistic endometriosis pain relationships
- Features: sleep, stress, activity, period phase, GI symptoms, medication, mood, hydration
- Transparent pain formula with known coefficients for validation

### Model Training
- **ElasticNet**: L1+L2 regularization, interpretable coefficients
- **RandomForest**: 300 trees, non-linear pattern capture
- Both models include prediction interval estimates
- 80/20 train/test split with consistent random seeds

### Model Registry (`src/registry.py`)
- Uniform interface for loading and using models
- Automatic feature ordering and validation
- Prediction clipping to 0-10 pain scale

## ⚠️ Important Disclaimer

This app is for **educational purposes only** and is not medical advice. Always consult with healthcare professionals for medical decisions.

## 🔧 Development

The app is designed to be modular and extensible. The ML models can be easily replaced with more sophisticated models trained on real data.

## 💜 Support

Built with love for the endometriosis community to help understand symptom patterns and management strategies.

