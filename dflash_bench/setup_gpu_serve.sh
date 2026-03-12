#!/usr/bin/env bash
set -euo pipefail

# setup_gpu_serve.sh — Build MLC LLM + TVM with CUDA, prepare models, and start serve.
# Uses full-precision target (q0f16) + bf16 draft (q0bf16) on CUDA.

cd ~/sky_workdir

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

echo "=== [1/7] Installing system dependencies ==="
sudo apt-get update -qq
sudo apt-get install -y -qq cmake ninja-build git python3-pip python3-venv libssl-dev
sudo apt-get install -y -qq llvm-15-dev
export LLVM_CONFIG=/usr/lib/llvm-15/bin/llvm-config

# Rust/cargo needed for xgrammar
if ! command -v cargo &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi
export PATH="$HOME/.cargo/bin:$PATH"

echo "=== [2/7] Setting up Python venv ==="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy decorator attrs psutil scipy packaging
pip install gradio openai

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
cmake .. -G Ninja
ninja -j$(nproc)
cd ~/sky_workdir

echo "=== [4/7] Building MLC LLM ==="
mkdir -p build && cd build
# TVM_SOURCE_DIR must be relative to project root (not build dir) since
# CMakeLists.txt does add_subdirectory(${TVM_SOURCE_DIR}).
# BUILD_DUMMY_LIBTVM OFF to link against the full TVM we just built.
cat > config.cmake <<'CMAKE_EOF'
set(USE_CUDA ON)
set(USE_CUBLAS ON)
set(USE_LLVM "/usr/lib/llvm-15/bin/llvm-config")
set(BUILD_DUMMY_LIBTVM OFF)
CMAKE_EOF
cmake .. -G Ninja
ninja -j$(nproc) mlc_llm_module
cd ~/sky_workdir

# Install tvm-ffi (has pyproject.toml with Cython build)
pip install -e 3rdparty/tvm/3rdparty/tvm-ffi
# TVM python has no setup.py — use PYTHONPATH instead
pip install -e python

export PYTHONPATH=$PWD/3rdparty/tvm/3rdparty/tvm-ffi/python:$PWD/3rdparty/tvm/python:$PWD/python:$PYTHONPATH
export TVM_LIBRARY_PATH=$PWD/3rdparty/tvm/build
export MLC_LLM_LIB_PATH=$PWD/build

echo "=== [5/7] Downloading and converting target model (q0f16) ==="
if [ ! -d "dist/Qwen3-8B-q0f16-MLC" ]; then
    python -m mlc_llm gen_config \
        Qwen/Qwen3-8B \
        --quantization q0f16 \
        --conv-template qwen2 \
        -o dist/Qwen3-8B-q0f16-MLC

    # Inject dflash_target_layer_ids
    python3 -c "
import json
p = 'dist/Qwen3-8B-q0f16-MLC/mlc-chat-config.json'
with open(p) as f: cfg = json.load(f)
cfg.setdefault('model_config', {})['dflash_target_layer_ids'] = [1, 9, 17, 25, 33]
with open(p, 'w') as f: json.dump(cfg, f, indent=2)
print('Injected dflash_target_layer_ids')
"

    python -m mlc_llm convert_weight \
        Qwen/Qwen3-8B \
        --quantization q0f16 \
        -o dist/Qwen3-8B-q0f16-MLC
else
    echo "Target model already converted."
fi

echo "=== [6/7] Downloading and converting draft model (q0bf16) ==="
if [ ! -d "dist/Qwen3-8B-DFlash-b16-q0bf16-MLC" ]; then
    python -m mlc_llm gen_config \
        z-lab/Qwen3-8B-DFlash-b16 \
        --quantization q0bf16 \
        --conv-template LM \
        --model-type dflash \
        -o dist/Qwen3-8B-DFlash-b16-q0bf16-MLC

    python -m mlc_llm convert_weight \
        z-lab/Qwen3-8B-DFlash-b16 \
        --quantization q0bf16 \
        -o dist/Qwen3-8B-DFlash-b16-q0bf16-MLC
else
    echo "Draft model already converted."
fi

echo "=== [7/7] Compiling model libraries ==="
# Compile target model for CUDA
if [ ! -f "dist/Qwen3-8B-q0f16-MLC/Qwen3-8B-q0f16-cuda.so" ]; then
    python -m mlc_llm compile \
        dist/Qwen3-8B-q0f16-MLC \
        --device cuda \
        -o dist/Qwen3-8B-q0f16-MLC/Qwen3-8B-q0f16-cuda.so
else
    echo "Target model lib already compiled."
fi

# Compile draft model for CUDA
if [ ! -f "dist/Qwen3-8B-DFlash-b16-q0bf16-MLC/Qwen3-8B-DFlash-b16-q0bf16-cuda.so" ]; then
    python compile_dflash.py \
        dist/Qwen3-8B-DFlash-b16-q0bf16-MLC \
        --quantization q0bf16 \
        --device cuda \
        --output dist/Qwen3-8B-DFlash-b16-q0bf16-MLC/Qwen3-8B-DFlash-b16-q0bf16-cuda.so
else
    echo "Draft model lib already compiled."
fi

echo "=== Setup complete ==="
ls -lh dist/*/*.so
