#!/usr/bin/env bash
# ============================================================
# 03_download_model.sh
# Run inside WSL2 after 02_setup_env.sh:
#
#   bash setup/03_download_model.sh
#
# Downloads deepseek-ai/DeepSeek-R1-Distill-Qwen-14B from
# Hugging Face into the default HF cache (~/.cache/huggingface).
#
# The model is publicly accessible; no HF token is required.
# If you later want a gated model, export HF_TOKEN=hf_xxx first.
#
# Storage: ~30 GB (BF16 safetensors)
# ============================================================

set -euo pipefail
GREEN="\033[0;32m"; CYAN="\033[0;36m"; YELLOW="\033[1;33m"; NC="\033[0m"
step()  { echo -e "\n${CYAN}[STEP]${NC} $*"; }
ok()    { echo -e "  ${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "  ${YELLOW}[WARN]${NC} $*"; }

MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
CONDA_DIR="$HOME/miniconda3"
CONDA_ENV="deepseek"

# Load HF_TOKEN from the project .env when it is not already exported.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

# Activate environment created by setup/02_setup_env.sh
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
    echo "Miniconda is not installed at $CONDA_DIR."
    echo "Run this first: bash setup/02_setup_env.sh"
    exit 1
fi
export PATH="$CONDA_DIR/bin:$PATH"
# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"
if ! conda env list | grep -q "^${CONDA_ENV} "; then
    echo "Conda environment '$CONDA_ENV' does not exist yet."
    echo "The model download depends on packages installed by the environment setup."
    echo "Run this first, then retry: bash setup/02_setup_env.sh"
    exit 1
fi
conda activate "$CONDA_ENV"

# ---- Check available disk space ------------------------------------------
step "Checking available disk space..."
AVAIL_GB=$(df -BG "$HOME" | awk 'NR==2 {gsub("G",""); print $4}')
echo "  Available in $HOME: ${AVAIL_GB} GB"
if [ "$AVAIL_GB" -lt 32 ]; then
    warn "Less than 32 GB free. Model weights are ~30 GB in BF16."
    warn "Consider pointing HF cache to a drive with more space:"
    warn "  export HF_HOME=/path/to/large/drive/.cache/huggingface"
    warn "Then re-run this script."
    read -r -p "Continue anyway? [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]] || exit 1
fi

# ---- Optionally set HF token ---------------------------------------------
if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    ok "HF token loaded. Downloads will use your Hugging Face account."
else
    ok "No HF_TOKEN set -- using public access (sufficient for this model)."
fi

# ---- Download model -------------------------------------------------------
step "Downloading $MODEL_ID ..."
echo "  This is ~30 GB and may take a while on first run."
echo "  Subsequent runs will use the local cache."

python - <<PYEOF
from huggingface_hub import snapshot_download
import os

model_id = "$MODEL_ID"
cache = snapshot_download(
    repo_id=model_id,
    ignore_patterns=["*.md", "*.txt", "original/*"],
)
print(f"  Model cached at: {cache}")
PYEOF

ok "Model downloaded."

# ---- Verify model can load (CPU meta check, no GPU needed) ---------------
step "Verifying model config is readable..."
python - <<PYEOF
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("$MODEL_ID")
print(f"  Architecture : {cfg.architectures}")
print(f"  Hidden size  : {cfg.hidden_size}")
print(f"  Num layers   : {cfg.num_hidden_layers}")
print(f"  Vocab size   : {cfg.vocab_size}")
PYEOF
ok "Config verified."

echo ""
echo -e "${GREEN}============================================================"
echo " Model download complete."
echo ""
echo " NEXT:"
echo "   Start the inference server: bash server/serve.sh"
echo "   Then run the smoke test   : python server/smoke_test.py"
echo "============================================================${NC}"
