#!/usr/bin/env bash
# ============================================================
# 02_setup_env.sh
# Run inside WSL2 Ubuntu-24.04 from the project root:
#
#   bash setup/02_setup_env.sh
#
# What it does:
#   1. Verifies NVIDIA GPU is visible (nvidia-smi)
#   2. Installs system build tools
#   3. Installs Miniconda (if not already present)
#   4. Creates a conda env "deepseek" with Python 3.11
#   5. Installs PyTorch 2.7 (CUDA 12.8 wheel) -- required for Blackwell/RTX-50xx
#   6. Installs vLLM, transformers, bitsandbytes, accelerate, and all project deps
# ============================================================

set -euo pipefail
GREEN="\033[0;32m"; CYAN="\033[0;36m"; YELLOW="\033[1;33m"; NC="\033[0m"
step()  { echo -e "\n${CYAN}[STEP]${NC} $*"; }
ok()    { echo -e "  ${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "  ${YELLOW}[WARN]${NC} $*"; }

CONDA_ENV="deepseek"
PYTHON_VER="3.11"

# ---- 1. GPU check ---------------------------------------------------------
step "Checking NVIDIA GPU visibility inside WSL..."
if ! command -v nvidia-smi &>/dev/null; then
    echo ""
    echo "nvidia-smi not found inside WSL."
    echo "Ensure you have:"
    echo "  - NVIDIA Windows driver >= 555 installed on the host (you have 610.88)"
    echo "  - WSL kernel updated: run 'wsl --update' in an elevated Windows PowerShell"
    echo "  - Launched this session with: wsl -d Ubuntu-24.04"
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
ok "GPU visible."

# ---- 2. System packages ---------------------------------------------------
step "Installing system build tools..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    build-essential git curl wget ca-certificates \
    libssl-dev libffi-dev python3-dev
ok "System packages ready."

# ---- 3. Miniconda ---------------------------------------------------------
step "Checking Miniconda..."
CONDA_DIR="$HOME/miniconda3"
if [ ! -d "$CONDA_DIR" ]; then
    step "Downloading and installing Miniconda..."
    INSTALLER="/tmp/miniconda.sh"
    wget -q "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -O "$INSTALLER"
    bash "$INSTALLER" -b -p "$CONDA_DIR"
    rm "$INSTALLER"
    ok "Miniconda installed at $CONDA_DIR"
else
    ok "Miniconda already present at $CONDA_DIR"
fi

# Add conda to PATH for this session
export PATH="$CONDA_DIR/bin:$PATH"
# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"

# Persist to .bashrc if not already there
if ! grep -q "miniconda3/bin" "$HOME/.bashrc" 2>/dev/null; then
    "$CONDA_DIR/bin/conda" init bash
    ok "conda added to .bashrc"
fi

# ---- 4. Conda environment -------------------------------------------------
step "Accepting Anaconda channel Terms of Service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
ok "Channel terms accepted."

step "Creating conda environment '$CONDA_ENV' (Python $PYTHON_VER)..."
if conda env list | grep -q "^${CONDA_ENV} "; then
    ok "Environment '$CONDA_ENV' already exists. Skipping creation."
else
    conda create -y -n "$CONDA_ENV" python="$PYTHON_VER"
    ok "Environment created."
fi

conda activate "$CONDA_ENV"
ok "Activated '$CONDA_ENV'."

# ---- 5. PyTorch with CUDA 12.8 (required for RTX 5080 / Blackwell) -------
step "Installing PyTorch 2.7 + CUDA 12.8..."
# RTX 5080 (GB203, Blackwell) requires PyTorch >= 2.6 and CUDA >= 12.6.
# Using the official cu128 index.
pip install --quiet \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available -- check driver and PyTorch install"
name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"  GPU: {name}  VRAM: {vram:.1f} GB")
EOF
ok "PyTorch CUDA verified."

# ---- 6. Inference and agent dependencies ----------------------------------
step "Installing inference stack and project dependencies..."
pip install \
    vllm \
    transformers \
    accelerate \
    bitsandbytes \
    huggingface_hub \
    sentencepiece \
    openai \
    pydantic \
    fastapi \
    uvicorn \
    httpx \
    pytest

ok "All packages installed."

# ---- Summary --------------------------------------------------------------
echo ""
echo -e "${GREEN}============================================================"
echo " Environment setup complete."
echo ""
echo " To activate: conda activate $CONDA_ENV"
echo ""
echo " NEXT STEPS:"
echo "   1. Run: bash setup/03_download_model.sh"
echo "   2. Then: bash server/serve.sh"
echo "============================================================${NC}"
