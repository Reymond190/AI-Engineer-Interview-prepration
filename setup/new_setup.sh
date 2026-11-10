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