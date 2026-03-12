#!/usr/bin/env bash
set -euo pipefail

# setup_gpu.sh — Build MLC LLM + TVM with CUDA and prepare models for benchmarking.
# Runs on the remote GPU instance. Expects repo at ~/mlc-llm.

cd ~/mlc-llm

# ── Environment ──
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
# Fix libstdc++ mismatch (miniconda's is too old for gcc-11 built TVM)
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

echo "=== [1/7] Installing system dependencies ==="
sudo apt-get update -qq
sudo apt-get install -y -qq \
  cmake ninja-build git python3-pip python3-venv libssl-dev \
  llvm-15-dev  # TVM requires LLVM 15+

# Rust/Cargo (needed by tokenizers-cpp)
if ! command -v cargo &>/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  export PATH=$HOME/.cargo/bin:$PATH
fi

echo "=== [2/7] Setting up Python venv ==="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -q \
  numpy decorator attrs psutil scipy packaging \
  fastapi uvicorn shortuuid requests \
  transformers huggingface_hub safetensors
pip install -q torch --index-url https://download.pytorch.org/whl/cu124

echo "=== [3/7] Building TVM with CUDA ==="
cd 3rdparty/tvm
mkdir -p build && cd build
cat > config.cmake <<'CMAKE_EOF'
set(USE_CUDA ON)
set(USE_CUBLAS ON)
set(USE_LLVM "/usr/lib/llvm-15/bin/llvm-config")
set(USE_RELAY_DEBUG OFF)
set(BUILD_DUMMY_LIBTVM OFF)
CMAKE_EOF
cmake .. -G Ninja 2>&1 | tail -5
ninja -j4
cd ~/mlc-llm

echo "=== [4/7] Building MLC LLM ==="
mkdir -p build && cd build
# Use BUILD_DUMMY_LIBTVM ON — links a minimal runtime lib.
# The full libtvm.so is loaded at runtime via TVM_LIBRARY_PATH.
cat > config.cmake <<'CMAKE_EOF'
set(USE_CUDA ON)
set(USE_CUBLAS ON)
set(USE_LLVM "/usr/lib/llvm-15/bin/llvm-config")
set(BUILD_DUMMY_LIBTVM ON)
CMAKE_EOF
cmake .. -G Ninja 2>&1 | tail -5
ninja -j4 mlc_llm_module
cd ~/mlc-llm

echo "=== [5/7] Installing Python packages ==="
# tvm-ffi needs _version.py before pip install (setuptools_scm can't generate it without .git)
TVM_FFI_DIR=3rdparty/tvm/3rdparty/tvm-ffi
python3 -c "
from pathlib import Path
v = Path('${TVM_FFI_DIR}/python/tvm_ffi/_version.py')
v.parent.mkdir(parents=True, exist_ok=True)
v.write_text('version = \"0.1.dev0\"\nversion_tuple = (0, 1, \"dev0\")\n')
print('Created tvm-ffi _version.py')
"
pip install -e "${TVM_FFI_DIR}" -q

# Symlink the full tvm_ffi.so from our TVM build into the installed package
TVM_FFI_LIB=$(python -c "import tvm_ffi; from pathlib import Path; print(Path(tvm_ffi.__file__).parent / 'lib')")
ln -sf "$PWD/3rdparty/tvm/build/lib/libtvm_ffi.so" "$TVM_FFI_LIB/libtvm_ffi.so"
echo "Symlinked libtvm_ffi.so → $TVM_FFI_LIB/"

# Set PYTHONPATH for TVM and MLC LLM (not pip-installable without issues)
export PYTHONPATH=$PWD/3rdparty/tvm/python:$PWD/python:${PYTHONPATH:-}
export TVM_LIBRARY_PATH=$PWD/3rdparty/tvm/build
export MLC_LLM_LIB_PATH=$PWD/build

# Verify imports
python -c "import tvm; print('TVM OK:', tvm.__file__); import mlc_llm; print('MLC LLM OK:', mlc_llm.__file__)"

echo "=== [6/7] Downloading and converting models ==="
mkdir -p dist

# Download models from HuggingFace
python3 -c "
from huggingface_hub import snapshot_download
print('Downloading Qwen3-8B...')
snapshot_download('Qwen/Qwen3-8B', local_dir='hf_models/Qwen3-8B')
print('Downloading Qwen3-8B-DFlash-b16...')
snapshot_download('z-lab/Qwen3-8B-DFlash-b16', local_dir='hf_models/Qwen3-8B-DFlash-b16')
print('Downloads complete.')
"

# Target model: Qwen3-8B → q0f16
if [ ! -f "dist/Qwen3-8B-q0f16-MLC/mlc-chat-config.json" ]; then
    python -m mlc_llm gen_config \
        hf_models/Qwen3-8B \
        --quantization q0f16 \
        --conv-template chatml \
        -o dist/Qwen3-8B-q0f16-MLC

    # Inject dflash_target_layer_ids into model config
    python3 -c "
import json
p = 'dist/Qwen3-8B-q0f16-MLC/mlc-chat-config.json'
with open(p) as f: cfg = json.load(f)
cfg.setdefault('model_config', {})['dflash_target_layer_ids'] = [1, 9, 17, 25, 33]
with open(p, 'w') as f: json.dump(cfg, f, indent=2)
print('Injected dflash_target_layer_ids')
"
else
    echo "Target config already exists, skipping gen_config."
fi

if [ ! -d "dist/Qwen3-8B-q0f16-MLC/params" ]; then
    python -m mlc_llm convert_weight \
        hf_models/Qwen3-8B \
        --quantization q0f16 \
        -o dist/Qwen3-8B-q0f16-MLC
else
    echo "Target weights already converted, skipping."
fi

# Draft model: Qwen3-8B-DFlash-b16 → q0f16
if [ ! -f "dist/Qwen3-8B-DFlash-b16-q0f16-MLC/mlc-chat-config.json" ]; then
    python -m mlc_llm gen_config \
        hf_models/Qwen3-8B-DFlash-b16 \
        --quantization q0f16 \
        --conv-template chatml \
        -o dist/Qwen3-8B-DFlash-b16-q0f16-MLC
else
    echo "Draft config already exists, skipping gen_config."
fi

if [ ! -d "dist/Qwen3-8B-DFlash-b16-q0f16-MLC/params" ]; then
    python -m mlc_llm convert_weight \
        hf_models/Qwen3-8B-DFlash-b16 \
        --quantization q0f16 \
        -o dist/Qwen3-8B-DFlash-b16-q0f16-MLC
else
    echo "Draft weights already converted, skipping."
fi

echo "=== [7/7] Compiling model libraries ==="
# Compile target model for CUDA
if [ ! -f "dist/Qwen3-8B-q0f16-MLC/Qwen3-8B-q0f16-cuda.so" ]; then
    python -m mlc_llm compile \
        dist/Qwen3-8B-q0f16-MLC \
        --device cuda \
        -o dist/Qwen3-8B-q0f16-MLC/Qwen3-8B-q0f16-cuda.so
else
    echo "Target model lib already compiled, skipping."
fi

# Compile draft model for CUDA
if [ ! -f "dist/Qwen3-8B-DFlash-b16-q0f16-MLC/Qwen3-8B-DFlash-b16-q0f16-cuda.so" ]; then
    python -m mlc_llm compile \
        dist/Qwen3-8B-DFlash-b16-q0f16-MLC \
        --device cuda \
        -o dist/Qwen3-8B-DFlash-b16-q0f16-MLC/Qwen3-8B-DFlash-b16-q0f16-cuda.so
else
    echo "Draft model lib already compiled, skipping."
fi

echo "=== Setup complete ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
ls -lh dist/*/
