#!/usr/bin/env bash
# Turn the Pi into a WiFi hotspot running the app in kiosk mode.
#
# Uses NetworkManager's built-in 'shared' AP (reboot-safe, no hostapd/dnsmasq).
# iptables DNAT 80 -> 8000 so users just open http://<pi-ip> and the captive
# portal probe (/generate_204 -> 204) auto-opens the app.
#
# Run after deploy/pi_setup.sh. Idempotent.
set -euo pipefail

SSID="${SSID:-PiLearnStation}"
WIFI_IF="${WIFI_IF:-wlan0}"
PORT="${PORT:-8000}"

say() { echo "==> $*"; }

say "Enabling hotspot on $WIFI_IF (SSID=$SSID)"
sudo nmcli device wifi hotspot ifname "$WIFI_IF" ssid "$SSID" password "learn1234"  2>/dev/null || {
  sudo nmcli connection delete "Hotspot" 2>/dev/null || true
  sudo nmcli device wifi hotspot ifname "$WIFI_IF" ssid "$SSID" password "learn1234"
}

IP="$(nmcli -g IP4.ADDRESS device show "$WIFI_IF" | awk -F/ '{print $1}')"
echo "   hotspot IP: $IP"

say "DNAT port 80 -> $PORT (captive portal hits both)"
sudo iptables -t nat -C PREROUTING -i "$WIFI_IF" -p tcp --dport 80 \
  -j REDIRECT --to-port "$PORT" 2>/dev/null || \
  sudo iptables -t nat -A PREROUTING -i "$WIFI_IF" -p tcp --dport 80 \
  -j REDIRECT --to-port "$PORT"

say "Installing systemd service (container) + kiosk autostart"
sudo tee /etc/systemd/system/pi-learn-station.service >/dev/null <<UNIT
[Unit]
Description=Pi Learning Station backend (container)
After=network-online.target
Wants=network-online.target

[Service]
User=$USER
ExecStart=/usr/bin/podman start -a pi-learn-station
ExecStop=/usr/bin/podman stop -t 10 pi-learn-station
ExecStartPost=/usr/bin/podman start pi-learn-station
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

# Ensure the container is created/known before the unit references it.
podman rm -f pi-learn-station 2>/dev/null || true
podman run -d --name pi-learn-station --restart=unless-stopped \
  -p "$PORT":8000 "${IMAGE:-pi-learn-station:arm64}"

# Kiosk: fullscreen Chromium on the hotspot IP, auto refresh.
KIO="$HOME/.config/autostart/pi-learn-kiosk.desktop"
mkdir -p "$(dirname "$KIO")"
cat > "$KIO" <<KIOSK
[Desktop Entry]
Type=Application
Name=Pi Learn Station Kiosk
Exec=chromium-browser --kiosk http://$IP --noerrdialogs --disable-infobars --autoplay-policy=no-user-gesture-required --ozone-platform=wayland
X-GNOME-Autostart-enabled=true
KIOSK

sudo systemctl daemon-reload
sudo systemctl enable --now pi-learn-station.service
echo
say "Done. Connect a phone to SSID '$SSID' -> http://$IP"