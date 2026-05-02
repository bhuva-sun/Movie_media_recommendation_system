#!/usr/bin/env bash
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Run as root: sudo bash $0"
  exit 1
fi

APP_DIR="/home/ubuntu/Movie-Recommendation-System-master"

if [[ ! -f "/etc/movie-reco.env" ]]; then
  echo "/etc/movie-reco.env is missing."
  echo "Create it using deploy/oracle/templates/movie-reco.env.template as a starting point."
  exit 1
fi

# Django setup
sudo -u ubuntu bash -lc "cd \"$APP_DIR\" && ./venv/bin/python manage.py migrate --noinput"
sudo -u ubuntu bash -lc "cd \"$APP_DIR\" && ./venv/bin/python manage.py collectstatic --noinput"

# systemd service
cp -f "$APP_DIR/deploy/oracle/templates/movie-reco.service" /etc/systemd/system/movie-reco.service
systemctl daemon-reload
systemctl enable --now movie-reco

# nginx config
cp -f "$APP_DIR/deploy/oracle/templates/nginx-movie-reco.conf" /etc/nginx/sites-available/movie-reco
ln -sf /etc/nginx/sites-available/movie-reco /etc/nginx/sites-enabled/movie-reco
rm -f /etc/nginx/sites-enabled/default || true
nginx -t
systemctl restart nginx

# firewall
if command -v ufw >/dev/null 2>&1; then
  ufw allow OpenSSH
  ufw allow 80/tcp
  ufw --force enable
fi

echo "Enabled services."
echo "Check:"
echo "  systemctl status movie-reco"
echo "  journalctl -u movie-reco -n 100 --no-pager"
echo "  curl -sS http://127.0.0.1/api/health/ || true"

