#!/usr/bin/env bash
set -euo pipefail

# bench_remote.sh — Run baseline and DFlash benchmarks on the remote GPU.
# Produces a comparison JSON in dflash_bench/results/.

cd ~/mlc-llm
source .venv/bin/activate
export PYTHONPATH=$PWD/3rdparty/tvm/python:$PWD/python:${PYTHONPATH:-}
export TVM_LIBRARY_PATH=$PWD/3rdparty/tvm/build
export MLC_LLM_LIB_PATH=$PWD/build
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

PORT=8787
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="dflash_bench/results"
mkdir -p "$RESULTS_DIR"

BASELINE_JSON="$RESULTS_DIR/baseline_${TIMESTAMP}.json"
DFLASH_JSON="$RESULTS_DIR/dflash_${TIMESTAMP}.json"
COMBINED_JSON="$RESULTS_DIR/${TIMESTAMP}.json"

# Model paths (bf16 target)
TARGET_DIR="dist/Qwen3-8B-q0f16-MLC"
TARGET_LIB="dist/Qwen3-8B-q0f16-MLC/Qwen3-8B-q0f16-cuda.so"
DRAFT_DIR="dist/Qwen3-8B-DFlash-b16-q0f16-MLC"
DRAFT_LIB="dist/Qwen3-8B-DFlash-b16-q0f16-MLC/Qwen3-8B-DFlash-b16-q0f16-cuda.so"

wait_for_server() {
    local max_wait=180
    local elapsed=0
    echo "  Waiting for server on port $PORT..."
    while [ $elapsed -lt $max_wait ]; do
        if curl -s "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then
            echo "  Server ready after ${elapsed}s"
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    echo "  ERROR: Server did not start within ${max_wait}s"
    return 1
}

kill_server() {
    echo "  Stopping server..."
    pkill -f "mlc_llm serve" 2>/dev/null || true
    sleep 2
    pkill -9 -f "mlc_llm serve" 2>/dev/null || true
    sleep 1
}

# Ensure no stale server is running
kill_server

# ─── Baseline benchmark ───
echo ""
echo "============================================================"
echo " Starting BASELINE server (no speculative decoding, bf16)"
echo "============================================================"

python -m mlc_llm serve \
    "$TARGET_DIR" \
    --model-lib "$TARGET_LIB" \
    --speculative-mode disable \
    --overrides "max_total_seq_length=2048;prefill_chunk_size=512" \
    --mode interactive --host 127.0.0.1 --port "$PORT" &
SERVER_PID=$!

wait_for_server

echo "  Running baseline benchmark..."
python bench_dflash.py baseline --port "$PORT" --output "$BASELINE_JSON"

kill_server
wait $SERVER_PID 2>/dev/null || true

# ─── DFlash benchmark ───
echo ""
echo "============================================================"
echo " Starting DFLASH server (speculative decoding, bf16)"
echo "============================================================"

python -m mlc_llm serve \
    "$TARGET_DIR" \
    --model-lib "$TARGET_LIB" \
    --speculative-mode dflash \
    --additional-models "${DRAFT_DIR},${DRAFT_LIB}" \
    --overrides "spec_draft_length=16;max_total_seq_length=2048;prefill_chunk_size=512" \
    --mode interactive --host 127.0.0.1 --port "$PORT" &
SERVER_PID=$!

wait_for_server

echo "  Running DFlash benchmark..."
python bench_dflash.py dflash --port "$PORT" --output "$DFLASH_JSON"

kill_server
wait $SERVER_PID 2>/dev/null || true

# ─── Combine results ───
echo ""
echo "============================================================"
echo " Combining results"
echo "============================================================"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)

python3 -c "
import json

with open('$BASELINE_JSON') as f:
    baseline = json.load(f)
with open('$DFLASH_JSON') as f:
    dflash = json.load(f)

baseline_tps = baseline['avg_tps']
dflash_tps = dflash['avg_tps']
speedup = dflash_tps / baseline_tps if baseline_tps > 0 else 0

combined = {
    'timestamp': '$TIMESTAMP',
    'gpu': '$GPU_NAME',
    'target_quantization': 'q0f16 (bf16)',
    'baseline': baseline,
    'dflash': dflash,
    'speedup': round(speedup, 3),
}

with open('$COMBINED_JSON', 'w') as f:
    json.dump(combined, f, indent=2)

print()
print('=' * 60)
print(' RESULTS SUMMARY')
print('=' * 60)
print(f'  GPU:       $GPU_NAME')
print(f'  Target:    q0f16 (bf16)')
print(f'  Baseline:  {baseline_tps:.1f} tok/s')
print(f'  DFlash:    {dflash_tps:.1f} tok/s')
print(f'  Speedup:   {speedup:.2f}x')
print(f'  Saved to:  $COMBINED_JSON')
print('=' * 60)
"

echo ""
echo "Benchmark complete. Results at: $COMBINED_JSON"
