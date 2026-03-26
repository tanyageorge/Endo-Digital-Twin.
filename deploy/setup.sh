#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Endo Digital Twin — AWS EC2 Ubuntu 22.04 Setup Script
# Run as: sudo bash deploy/setup.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

APP_USER="ubuntu"
APP_DIR="/home/${APP_USER}/Endo-Digital-Twin"
VENV_DIR="${APP_DIR}/venv"
SERVICE_NAME="endo-digital-twin"
NGINX_CONF="/etc/nginx/sites-available/${SERVICE_NAME}"
PORT=8501

echo "======================================================"
echo " Endo Digital Twin — Server Setup"
echo "======================================================"

# ── 1. System packages ──────────────────────────────────
echo "[1/7] Installing system packages..."
apt-get update -qq
apt-get install -y -qq \
    python3.11 \
    python3.11-venv \
    python3-pip \
    nginx \
    curl \
    git \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev

# ── 2. Python virtualenv ────────────────────────────────
echo "[2/7] Creating Python virtualenv..."
if [ ! -d "${VENV_DIR}" ]; then
    sudo -u "${APP_USER}" python3.11 -m venv "${VENV_DIR}"
fi

# ── 3. Install Python dependencies ─────────────────────
echo "[3/7] Installing Python dependencies..."
sudo -u "${APP_USER}" "${VENV_DIR}/bin/pip" install --upgrade pip --quiet
sudo -u "${APP_USER}" "${VENV_DIR}/bin/pip" install -r "${APP_DIR}/requirements.txt" --quiet

# ── 4. Generate data and train models ──────────────────
echo "[4/7] Generating synthetic data and training ML models..."
cd "${APP_DIR}"
sudo -u "${APP_USER}" "${VENV_DIR}/bin/python" src/synth.py
sudo -u "${APP_USER}" "${VENV_DIR}/bin/python" src/train_elasticnet.py
sudo -u "${APP_USER}" "${VENV_DIR}/bin/python" src/train_rf.py

# ── 5. Install systemd service ──────────────────────────
echo "[5/7] Installing systemd service..."
cp "${APP_DIR}/deploy/${SERVICE_NAME}.service" "/etc/systemd/system/${SERVICE_NAME}.service"

# Substitute actual app directory path into service file
sed -i "s|/home/ubuntu/Endo-Digital-Twin|${APP_DIR}|g" "/etc/systemd/system/${SERVICE_NAME}.service"

systemctl daemon-reload
systemctl enable "${SERVICE_NAME}"
systemctl start "${SERVICE_NAME}"

echo "Service status:"
systemctl status "${SERVICE_NAME}" --no-pager || true

# ── 6. Configure nginx ──────────────────────────────────
echo "[6/7] Configuring nginx reverse proxy..."
cp "${APP_DIR}/deploy/nginx.conf" "${NGINX_CONF}"

# Remove default nginx site if it exists
if [ -L /etc/nginx/sites-enabled/default ]; then
    rm /etc/nginx/sites-enabled/default
fi

# Enable our site
ln -sf "${NGINX_CONF}" "/etc/nginx/sites-enabled/${SERVICE_NAME}"

nginx -t
systemctl enable nginx
systemctl restart nginx

# ── 7. Firewall ─────────────────────────────────────────
echo "[7/7] Configuring firewall..."
if command -v ufw &>/dev/null; then
    ufw allow 80/tcp
    ufw allow 22/tcp
    ufw --force enable || true
fi

echo ""
echo "======================================================"
echo " Setup complete!"
echo "======================================================"
echo ""
echo " App running at:  http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo '<EC2-PUBLIC-IP>')"
echo " Service logs:    sudo journalctl -u ${SERVICE_NAME} -f"
echo " Restart app:     sudo systemctl restart ${SERVICE_NAME}"
echo ""
echo " To enable Claude AI insights, set your API key:"
echo "   sudo systemctl edit ${SERVICE_NAME}"
echo "   # Add under [Service]: Environment=\"ANTHROPIC_API_KEY=sk-ant-...\""
echo "   sudo systemctl daemon-reload && sudo systemctl restart ${SERVICE_NAME}"
echo ""
