#!/usr/bin/env bash
# ============================================================
# server/serve.sh
# Start the vLLM OpenAI-compatible server.
#
# Usage (inside WSL from project root):
#   bash server/serve.sh            # Qwen3.5-9B (default, agent work)
#   bash server/serve.sh qwen3      # Qwen3-8B (faster thinking, weaker tool use)
#   bash server/serve.sh deepseek   # DeepSeek-R1-Distill-Qwen-14B (old setup)
#   bash server/serve.sh ft         # Qwen3-0.6B + finetune_course LoRA adapters (lesson 13)
#
# The server listens on http://localhost:8000 and is compatible
# with the OpenAI Python SDK.
#
# Profiles (RTX 5080, 16 GB):
#   qwen3    Qwen/Qwen3-8B-AWQ, ~6 GB weights. Native tool calling
#            (hermes parser) and hybrid thinking: the reasoning parser
#            moves <think> into `reasoning_content`, and clients can turn
#            thinking off per request with
#            chat_template_kwargs={"enable_thinking": false}.
#            32k context, 4 concurrent sequences (parallel agents).
#   qwen35   RedHatAI/Qwen3.5-9B-quantized.w4a16 (INT4 weights, compressed-
#            tensors), ~11 GB on disk; the 248k-vocab embeddings stay BF16.
#            Hybrid Gated-DeltaNet/attention layers -> small KV cache.
#            Vision-language model; --language-model-only skips the
#            vision encoder. Tool calls use the qwen3_coder (XML) parser.
#   deepseek casperhansen/deepseek-r1-distill-qwen-14b-awq, ~10 GB.
#            No tool calling, 8k context, 1 sequence.
#
# vLLM >= 0.32 no longer supports bitsandbytes, so all profiles use
# pre-quantized checkpoints. Weights download on first start.
# ============================================================

set -euo pipefail
CYAN="\033[0;36m"; GREEN="\033[0;32m"; NC="\033[0m"

CONDA_DIR="$HOME/miniconda3"
CONDA_ENV="deepseek"
PORT=8000
TENSOR_PARALLEL=1   # single GPU
GPU_MEM_UTIL=0.85   # leave headroom for the Windows desktop's VRAM use
PROFILE="${1:-qwen35}"

# JSON-schema guided decoding otherwise allows unlimited whitespace, and after
# a long <think> the model can get stuck emitting newlines forever.
STRUCTURED_ARGS=(
    --structured-outputs-config.backend xgrammar
    --structured-outputs-config.disable_any_whitespace true
)

case "$PROFILE" in
    qwen3)
        MODEL="Qwen/Qwen3-8B-AWQ"
        SERVED_NAME="Qwen/Qwen3-8B"   # name clients request
        MAX_LEN=32768
        MAX_SEQS=4
        QUANT_ARGS=(--quantization awq_marlin --dtype float16)
        EXTRA_ARGS=(
            --enable-auto-tool-choice
            --tool-call-parser hermes
            --reasoning-parser qwen3
            "${STRUCTURED_ARGS[@]}"
        )
        ;;
    qwen35)
        MODEL="RedHatAI/Qwen3.5-9B-quantized.w4a16"
        SERVED_NAME="Qwen/Qwen3.5-9B"
        MAX_LEN=32768
        MAX_SEQS=4
        QUANT_ARGS=()   # compressed-tensors config is read from the checkpoint
        EXTRA_ARGS=(
            --language-model-only
            --enable-auto-tool-choice
            --tool-call-parser qwen3_coder
            --reasoning-parser qwen3
            "${STRUCTURED_ARGS[@]}"
        )
        ;;
    ft)
        # finetune_course lesson 13: one base, two LoRA adapters, chosen per
        # request with model="tickets" | "toolcalls" | "Qwen/Qwen3-0.6B".
        ADAPTERS="$(dirname "$(realpath "$0")")/../finetune_course/outputs"
        MODEL="Qwen/Qwen3-0.6B"
        SERVED_NAME="Qwen/Qwen3-0.6B"
        MAX_LEN=4096
        MAX_SEQS=16
        QUANT_ARGS=()
        EXTRA_ARGS=(
            --enable-lora
            --max-lora-rank 64
            --max-loras 2
            --lora-modules "tickets=$ADAPTERS/05_sft_lora" "toolcalls=$ADAPTERS/11_tool_lora"
            --enable-auto-tool-choice
            --tool-call-parser hermes
            --reasoning-parser qwen3
        )
        ;;
    deepseek)
        MODEL="casperhansen/deepseek-r1-distill-qwen-14b-awq"
        SERVED_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
        MAX_LEN=8192
        MAX_SEQS=1
        QUANT_ARGS=(--quantization awq_marlin --dtype float16)
        EXTRA_ARGS=()
        ;;
    *)
        echo "Unknown profile '$PROFILE' (expected: qwen3 | qwen35 | deepseek | ft)" >&2
        exit 1
        ;;
esac

# FlashInfer's sampler JIT-compiles with nvcc, which isn't installed here;
# use vLLM's built-in PyTorch sampler instead.
export VLLM_USE_FLASHINFER_SAMPLER=0

# Load .env if present
ENV_FILE="$(dirname "$(realpath "$0")")/../.env"
if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    set -o allexport; source "$ENV_FILE"; set +o allexport
fi

# Activate conda
export PATH="$CONDA_DIR/bin:$PATH"
# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo -e "${CYAN}[serve]${NC} Starting vLLM server (profile: $PROFILE)..."
echo "  Model          : $MODEL"
echo "  Quantization   : ${QUANT_ARGS[*]:-from checkpoint config}"
echo "  Max ctx length : $MAX_LEN tokens"
echo "  Max sequences  : $MAX_SEQS"
echo "  Port           : $PORT"
echo ""
echo "  API base       : http://localhost:$PORT/v1"
echo "  Model name     : $SERVED_NAME"
echo ""
echo "  Press Ctrl+C to stop."
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model              "$MODEL" \
    "${QUANT_ARGS[@]}" \
    --max-model-len      "$MAX_LEN" \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --max-num-seqs       "$MAX_SEQS" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --port               "$PORT" \
    --host               0.0.0.0 \
    --served-model-name  "$SERVED_NAME" \
    --disable-log-stats \
    "${EXTRA_ARGS[@]}"

echo -e "${GREEN}[serve]${NC} Server stopped."
