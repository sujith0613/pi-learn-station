#!/usr/bin/env bash
# One-time setup on the Raspberry Pi: podman, load the arm64 image, run the app.
# Run as the Pi user. Idempotent.
#
# The image tarball (pi-learn-station-arm64.tar.gz) is produced on the build
# machine with:  podman save pi-learn-station:arm64 | gzip > pi-learn-station-arm64.tar.gz
# Copy it next to this repo, then run this script.
set -euo pipefail

REPO="${REPO:-$HOME/pi-learn-station}"
TARBALL="${TARBALL:-$HOME/pi-learn-station-arm64.tar.gz}"
IMAGE="${IMAGE:-pi-learn-station:arm64}"
PORT="${PORT:-8000}"

say() { echo "==> $*"; }

say "Installing podman (container runtime for the app)"
command -v podman >/dev/null 2>&1 || sudo apt-get update -qq && \
  sudo apt-get install -y -qq podman

say "Installing hotspot deps (NetworkManager AP mode, Chromium kiosk)"
sudo apt-get install -y -qq network-manager chromium-browser

say "Loading container image from $TARBALL"
if [ ! -f "$TARBALL" ]; then
  echo "ERROR: image tarball not found at $TARBALL"
  echo "  On the build machine run: podman save $IMAGE | gzip > $TARBALL"
  exit 1
fi
podman load -i "$TARBALL"

say "Starting container on 0.0.0.0:$PORT (restart=unless-stopped)"
podman rm -f pi-learn-station 2>/dev/null || true
podman run -d --name pi-learn-station --restart=unless-stopped \
  -p "$PORT":8000 "$IMAGE"

say "Verifying health"
for i in $(seq 1 10); do
  if curl -fsS "http://localhost:$PORT/api/health" | grep -q '"status":"ok"'; then
    say "OK: app healthy on port $PORT"
    break
  fi
  [ "$i" -eq 10 ] && { echo "ERROR: app did not become healthy"; exit 1; }
  sleep 2
done

echo
echo "Next: run scripts/deploy_pi_hotspot.sh to enable AP mode + autostart."
