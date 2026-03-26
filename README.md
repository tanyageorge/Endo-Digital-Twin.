# Endo Digital Twin — Clinical AI

> A production-quality ML web application for endometriosis pain prediction with what-if lifestyle simulation, SHAP feature attribution, and Claude AI personalised insights.

**Live App:** `http://<your-ec2-public-ip>` — deployed on AWS EC2, running 24/7.

---

## Features

| Feature | Description |
|---|---|
| **Digital Twin Simulator** | Adjust 8 lifestyle variables (sleep, stress, activity, hydration, mood, cycle phase, GI, NSAID) and instantly see predicted pain |
| **Dual ML Models** | ElasticNet (interpretable coefficients) and RandomForest (non-linear patterns) |
| **SHAP Waterfall Chart** | Per-feature contribution to pain change visualised as a force plot |
| **Pain Landscape Heatmap** | 2D grid showing how any two feature pairs interact across their full range |
| **Well-being Gauge** | Weighted lifestyle balance score (0–100) |
| **Claude AI Insights** | Describe symptoms in plain English → Claude reads your simulation inputs and returns personalised, evidence-based recommendations |
| **Symptom Tracker** | Daily check-in log with pain timeline, distribution, sleep vs pain scatter |
| **Dark Clinical UI** | Glassmorphism design, gradient accents, Space Grotesk font — looks like a real clinical AI product |

---

## ML Pipeline

### Data Generation (`src/synth.py`)
- 8,000 synthetic samples with clinically-informed pain formula
- Features: `sleep`, `stress`, `activity`, `period_phase`, `gi`, `meds`, `mood`, `hydration`
- Transparent formula with known coefficients for ground-truth validation

### Models
| Model | Training | Interpretation |
|---|---|---|
| **ElasticNet** | StandardScaler + ElasticNet(α=0.12, L1_ratio=0.2) | Exact coefficients → exact SHAP |
| **RandomForest** | 300 trees, max_depth=10, min_samples_leaf=2 | Feature importances → SHAP estimates |

Both models store `pi_halfwidth` (80th-percentile residual) as a prediction interval estimate.

---

## Quick Start (Local)

```bash
# 1. Clone the repo
git clone https://github.com/tanyageorge/Endo-Digital-Twin.git
cd Endo-Digital-Twin

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate synthetic data and train models
python src/synth.py
python src/train_elasticnet.py
python src/train_rf.py

# 4. (Optional) Set Anthropic API key for Claude AI features
export ANTHROPIC_API_KEY=sk-ant-...

# 5. Run the app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Project Structure

```
Endo-Digital-Twin/
├── app.py                      # Main Streamlit application (dark UI, Claude, SHAP)
├── requirements.txt            # Python dependencies
├── README.md
├── checkin_data.csv            # User tracking data (auto-created)
├── women_endo.png              # Hero image
├── .streamlit/
│   └── config.toml             # Dark theme configuration
├── data/
│   └── synthetic.csv           # Generated training data
├── models/
│   ├── model_en.pkl            # Trained ElasticNet
│   └── model_rf.pkl            # Trained RandomForest
├── src/
│   ├── synth.py                # Synthetic data generator
│   ├── train_elasticnet.py     # ElasticNet training pipeline
│   ├── train_rf.py             # RandomForest training pipeline
│   ├── registry.py             # Model loader / uniform predict interface
│   ├── explain.py              # Explanation utilities
│   ├── model.py                # Legacy mock model (unused)
│   └── visualization.py       # Chart utilities (legacy, superseded by app.py)
└── deploy/
    ├── setup.sh                # AWS EC2 Ubuntu setup script
    ├── endo-digital-twin.service  # systemd service file
    └── nginx.conf              # Nginx reverse proxy config
```

---

## AWS EC2 Deployment

### Prerequisites
- AWS EC2 instance: Ubuntu 22.04 LTS, t2.micro or larger
- Port 80 (HTTP) and 22 (SSH) open in Security Group
- Your Anthropic API key

### Step 1 — Launch EC2 and SSH in

```bash
ssh -i your-key.pem ubuntu@<EC2-PUBLIC-IP>
```

### Step 2 — Upload the app

```bash
# From your local machine
scp -i your-key.pem -r ./Endo-Digital-Twin ubuntu@<EC2-PUBLIC-IP>:/home/ubuntu/
```

Or clone directly on EC2:
```bash
git clone https://github.com/tanyageorge/Endo-Digital-Twin.git
```

### Step 3 — Run the setup script

```bash
cd /home/ubuntu/Endo-Digital-Twin
chmod +x deploy/setup.sh
sudo deploy/setup.sh
```

This script:
- Installs Python 3.11, pip, nginx
- Creates a Python virtualenv
- Installs all pip dependencies
- Trains the ML models
- Installs and enables the systemd service
- Configures nginx as reverse proxy on port 80

### Step 4 — Set your API key

```bash
sudo systemctl edit endo-digital-twin
```
Add under `[Service]`:
```
Environment="ANTHROPIC_API_KEY=sk-ant-..."
```
Then:
```bash
sudo systemctl daemon-reload
sudo systemctl restart endo-digital-twin
```

### Step 5 — Access the app

Open `http://<EC2-PUBLIC-IP>` in your browser. The app runs on port 80 via nginx, 24/7, auto-restarts on reboot.

### Useful Commands

```bash
sudo systemctl status endo-digital-twin    # Check service status
sudo journalctl -u endo-digital-twin -f    # Tail logs
sudo systemctl restart endo-digital-twin   # Restart app
sudo nginx -t && sudo systemctl reload nginx  # Reload nginx config
```

---

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude AI features | Optional (AI tab disabled without it) |

---

## Disclaimer

This application is for **educational purposes only** and does not constitute medical advice. Models are trained on synthetic data. Always consult a qualified healthcare professional for medical decisions related to endometriosis.

---

## Tech Stack

- **Frontend:** Streamlit, Plotly, custom CSS (glassmorphism dark theme, Space Grotesk font)
- **ML:** scikit-learn (ElasticNet, RandomForest), joblib, numpy, pandas
- **AI:** Anthropic Claude API (`claude-opus-4-6`)
- **Explainability:** SHAP-style waterfall chart (Plotly Waterfall, coefficient-based)
- **Deployment:** AWS EC2, nginx, systemd, Python virtualenv

---

Built with care for the endometriosis community.
