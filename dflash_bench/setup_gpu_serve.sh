#!/usr/bin/env bash
set -euo pipefail

# setup_gpu_serve.sh — Build MLC LLM + TVM with CUDA, prepare models, and start serve.
# Uses full-precision target (q0f16) + bf16 draft (q0bf16) on CUDA.

cd ~/sky_workdir

echo "=== [1/7] Installing system dependencies ==="
sudo apt-get update -qq
sudo apt-get install -y -qq cmake ninja-build git python3-pip python3-venv libssl-dev

# Install CUDA toolkit if nvcc is not found (RunPod has driver only)
if ! command -v nvcc &> /dev/null && [ ! -f /usr/local/cuda/bin/nvcc ]; then
    echo "  Installing CUDA toolkit..."
    # Use the NVIDIA CUDA keyring for apt
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update -qq
    sudo apt-get install -y -qq cuda-toolkit-12-4
    rm -f cuda-keyring_1.1-1_all.deb
fi
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

# Install LLVM (try 15 first, fall back to whatever is available)
if ! dpkg -s llvm-15-dev &>/dev/null 2>&1; then
    sudo apt-get install -y -qq llvm-15-dev 2>/dev/null || sudo apt-get install -y -qq llvm-dev
fi
if [ -f /usr/lib/llvm-15/bin/llvm-config ]; then
    export LLVM_CONFIG=/usr/lib/llvm-15/bin/llvm-config
else
    export LLVM_CONFIG=$(which llvm-config || echo "/usr/bin/llvm-config")
fi

# Rust/cargo needed for xgrammar
if ! command -v cargo &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi
export PATH="$HOME/.cargo/bin:$PATH"

# Fix libstdc++ for miniconda environments
if [ -f /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ]; then
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
fi

echo "=== [2/7] Setting up Python venv ==="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy decorator attrs psutil scipy packaging
pip install gradio openai huggingface_hub

echo "=== [3/7] Building TVM with CUDA ==="
cd 3rdparty/tvm
mkdir -p build && cd build
cat > config.cmake <<CMAKE_EOF
set(USE_CUDA ON)
set(USE_CUBLAS ON)
set(USE_LLVM "${LLVM_CONFIG}")
set(USE_RELAY_DEBUG OFF)
set(BUILD_DUMMY_LIBTVM OFF)
CMAKE_EOF
cmake .. -G Ninja
ninja -j$(( $(nproc) / 2 ))
cd ~/sky_workdir

echo "=== [4/7] Building MLC LLM ==="
mkdir -p build && cd build
cat > config.cmake <<CMAKE_EOF
set(USE_CUDA ON)
set(USE_CUBLAS ON)
set(USE_LLVM "${LLVM_CONFIG}")
set(BUILD_DUMMY_LIBTVM OFF)
CMAKE_EOF
cmake .. -G Ninja
ninja -j$(( $(nproc) / 2 )) mlc_llm_module
cd ~/sky_workdir

# tvm-ffi editable install fails in SkyPilot workdir (no .git for setuptools_scm).
# Create the _version.py manually and install without editable mode.
mkdir -p 3rdparty/tvm/3rdparty/tvm-ffi/python/tvm_ffi
echo '__version__ = "0.0.0"' > 3rdparty/tvm/3rdparty/tvm-ffi/python/tvm_ffi/_version.py
pip install -e 3rdparty/tvm/3rdparty/tvm-ffi
# MLC LLM python package
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
        --opt "flashinfer=1;cublas_gemm=0;faster_transformer=0;cudagraph=1;cutlass=1" \
        --output dist/Qwen3-8B-DFlash-b16-q0bf16-MLC/Qwen3-8B-DFlash-b16-q0bf16-cuda.so
else
    echo "Draft model lib already compiled."
fi

echo "=== Setup complete ==="
ls -lh dist/*/*.so
