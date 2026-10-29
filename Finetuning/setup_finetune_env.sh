#!/usr/bin/env bash
# ============================================================
# 05_setup_finetune_env.sh
# Run inside WSL2 Ubuntu-24.04 from the project root (after 02_setup_env.sh):
#
#   bash setup/05_setup_finetune_env.sh
#
# Creates a SEPARATE conda env "ft" for finetune_course/. vLLM pins an exact
# torch version in the "deepseek" env; training libraries (trl, peft,
# bitsandbytes, liger-kernel, torchao) move fast and would fight those pins.
# Two envs = upgrading one never breaks the other.
#
# What it does:
#   1. Creates conda env "ft" (Python 3.12)
#   2. Installs PyTorch (CUDA wheel; Blackwell / RTX 50xx needs >= cu128)
#   3. Installs the HF training stack + efficiency libraries
#   4. Tries optional extras (flash-attn, galore) without failing
#   5. Sanity checks: GPU visible, bf16, 4-bit (bitsandbytes) matmul works
# ============================================================

set -euo pipefail
GREEN="\033[0;32m"; CYAN="\033[0;36m"; YELLOW="\033[1;33m"; NC="\033[0m"
step()  { echo -e "\n${CYAN}[STEP]${NC} $*"; }
ok()    { echo -e "  ${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "  ${YELLOW}[WARN]${NC} $*"; }

CONDA_ENV="ft"
PYTHON_VER="3.12"
CONDA_DIR="$HOME/miniconda3"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"

export PATH="$CONDA_DIR/bin:$PATH"
# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"

# ---- 1. Conda environment ---------------------------------------------------
step "Creating conda environment '$CONDA_ENV' (Python $PYTHON_VER)..."
if conda env list | grep -q "^${CONDA_ENV} "; then
    ok "Environment '$CONDA_ENV' already exists. Skipping creation."
else
    conda create -y -n "$CONDA_ENV" python="$PYTHON_VER"
    ok "Environment created."
fi
conda activate "$CONDA_ENV"

# ---- 2. PyTorch -------------------------------------------------------------
step "Installing PyTorch from $TORCH_INDEX ..."
pip install --quiet --upgrade pip
pip install --quiet torch --index-url "$TORCH_INDEX"
ok "PyTorch installed."

# ---- 3. Training stack ------------------------------------------------------
step "Installing transformers / peft / trl / datasets / accelerate / bitsandbytes..."
pip install --quiet \
    transformers peft trl datasets accelerate bitsandbytes \
    liger-kernel torchao \
    python-dotenv matplotlib
ok "Training stack installed."

# ---- 4. Optional extras (lessons skip them if missing) ----------------------
step "Trying optional extras..."
pip install --quiet galore-torch && ok "galore-torch installed." || warn "galore-torch failed (lesson 16 skips GaLore)."
pip install --quiet apollo-torch && ok "apollo-torch installed." || warn "apollo-torch failed (lesson 16 skips APOLLO)."
# flash-attn has no official Blackwell wheel for every torch version and a
# source build needs nvcc + ~1 h. Lessons fall back to PyTorch SDPA, which
# already uses a FlashAttention-style kernel.
pip install --quiet --no-build-isolation flash-attn --only-binary :all: 2>/dev/null \
    && ok "flash-attn installed." || warn "flash-attn wheel unavailable (SDPA is used instead)."

# ---- 5. Sanity checks -------------------------------------------------------
step "Sanity checks..."
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available"
p = torch.cuda.get_device_properties(0)
print(f"  GPU: {p.name}  {p.total_memory / 1024**3:.1f} GB  sm_{p.major}{p.minor}  bf16={torch.cuda.is_bf16_supported()}")

import bitsandbytes as bnb
import torch.nn as nn
lin = bnb.nn.Linear4bit(64, 64, compute_dtype=torch.bfloat16, quant_type="nf4")
lin = lin.cuda()
y = lin(torch.randn(2, 64, device="cuda", dtype=torch.bfloat16))
print(f"  bitsandbytes {bnb.__version__}: 4-bit NF4 matmul OK, out={tuple(y.shape)}")

import importlib
for m in ["transformers", "peft", "trl", "datasets", "accelerate", "liger_kernel", "torchao", "galore_torch", "apollo_torch", "flash_attn"]:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:13s} {getattr(mod, '__version__', 'ok')}")
    except Exception as e:  # noqa: BLE001  report, don't fail, on optional extras
        print(f"  {m:13s} MISSING ({type(e).__name__})")
EOF
ok "Done. Activate with: conda activate $CONDA_ENV"
