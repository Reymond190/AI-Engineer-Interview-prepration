#!/usr/bin/env bash
# ============================================================
# setup/04_install_docker.sh
# Install Docker Engine (not Docker Desktop) inside WSL Ubuntu 24.04,
# from Docker's official apt repository.
#
# Usage (inside WSL, from project root):
#   bash setup/04_install_docker.sh
# You'll be asked for your sudo password. Afterwards, close and reopen
# the WSL terminal (or run `newgrp docker`) so `docker` works without sudo.
# ============================================================

set -euo pipefail

if command -v docker >/dev/null 2>&1; then
    echo "[docker] already installed: $(docker --version)"
    exit 0
fi

echo "[docker] adding Docker's apt repository..."
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null

echo "[docker] installing engine..."
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "[docker] enabling service and adding $USER to the docker group..."
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"

echo "[docker] smoke test..."
sudo docker run --rm hello-world | head -3

echo
echo "[docker] done. Reopen the WSL terminal (or run: newgrp docker), then check: docker ps"
