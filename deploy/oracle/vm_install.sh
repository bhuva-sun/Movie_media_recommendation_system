#!/usr/bin/env bash
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Run as root: sudo bash $0"
  exit 1
fi

APP_DIR="/home/ubuntu/Movie-Recommendation-System-master"

apt update
apt install -y python3 python3-venv python3-pip nginx unzip

if [[ ! -d "$APP_DIR" ]]; then
  # If user only uploaded zip
  if [[ -f "/home/ubuntu/movie-reco.zip" ]]; then
    sudo -u ubuntu bash -lc "cd /home/ubuntu && unzip -o movie-reco.zip -d Movie-Recommendation-System-master"
  fi
fi

if [[ ! -f "$APP_DIR/manage.py" ]]; then
  echo "manage.py not found at $APP_DIR. Ensure the project is uploaded/unzipped correctly."
  exit 1
fi

sudo -u ubuntu bash -lc "cd \"$APP_DIR\" && python3 -m venv venv"
sudo -u ubuntu bash -lc "cd \"$APP_DIR\" && ./venv/bin/python -m pip install --upgrade pip"
sudo -u ubuntu bash -lc "cd \"$APP_DIR\" && ./venv/bin/python -m pip install -r requirements.txt"

# Prepare staticfiles directory (Nginx alias target)
sudo -u ubuntu bash -lc "mkdir -p \"$APP_DIR/staticfiles\""

echo "VM dependencies installed. Next: configure /etc/movie-reco.env and enable services."

