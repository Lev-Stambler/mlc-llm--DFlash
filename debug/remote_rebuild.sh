#!/usr/bin/env bash
# remote_rebuild.sh — Fast incremental C++ rebuild on remote GPU instance.
#
# First run: installs ccache, does full build.
# Subsequent runs: rsyncs changed C++ files, rebuilds only what changed (~3-5s).
#
# Usage:
#   ./debug/remote_rebuild.sh <host>                    # rsync + rebuild
#   ./debug/remote_rebuild.sh <host> --setup            # first-time ccache install
#   ./debug/remote_rebuild.sh <host> --test "command"   # rebuild then run command
#
# Examples:
#   ./debug/remote_rebuild.sh 100.31.202.84 --setup
#   ./debug/remote_rebuild.sh 100.31.202.84
#   ./debug/remote_rebuild.sh 100.31.202.84 --test "python test_engine_init.py"
#   REMOTE_DIR=~/mlc-llm ./debug/remote_rebuild.sh gpu-box

set -euo pipefail

HOST="${1:?Usage: $0 <host> [--setup] [--test \"command\"]}"
shift

REMOTE_DIR="${REMOTE_DIR:-~/sky_workdir}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SETUP=false
TEST_CMD=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --setup) SETUP=true; shift ;;
    --test)  TEST_CMD="$2"; shift 2 ;;
    *)       echo "Unknown flag: $1"; exit 1 ;;
  esac
done

# ── Step 1: First-time ccache setup ──
if $SETUP; then
  echo "=== Installing ccache on $HOST ==="
  ssh "$HOST" bash -s "$REMOTE_DIR" <<'REMOTE_SETUP'
    set -euo pipefail
    REMOTE_DIR="$1"

    # Install ccache
    if ! command -v ccache &>/dev/null; then
      echo "Installing ccache..."
      sudo apt-get update -qq
      sudo apt-get install -y -qq ccache
    fi
    echo "ccache: $(ccache --version | head -1)"

    # Reconfigure cmake to use ccache
    cd "$REMOTE_DIR"
    if [ -f build/CMakeCache.txt ]; then
      echo "Reconfiguring build with ccache..."
      cd build
      cmake .. -G Ninja \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
      echo "Build reconfigured with ccache."
    else
      echo "WARNING: No existing build found at $REMOTE_DIR/build"
      echo "Run setup_gpu_serve.sh first, then re-run with --setup"
    fi

    # Also reconfigure TVM build with ccache
    cd "$REMOTE_DIR"
    if [ -f 3rdparty/tvm/build/CMakeCache.txt ]; then
      echo "Reconfiguring TVM build with ccache..."
      cd 3rdparty/tvm/build
      cmake .. -G Ninja \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
      echo "TVM build reconfigured with ccache."
    fi

    # Prime ccache with a build so future rebuilds are fast
    echo "Priming ccache with initial build..."
    cd "$REMOTE_DIR"
    cmake --build build --target mlc_llm_module -j$(nproc)
    echo ""
    echo "ccache stats after priming:"
    ccache -s 2>/dev/null | grep -E "(Hits|Misses|Cache size)" || true
REMOTE_SETUP
  echo "=== ccache setup complete ==="
fi

# ── Step 2: Rsync changed C++ source files ──
echo "=== Syncing C++ sources to $HOST:$REMOTE_DIR ==="
START=$(date +%s%3N 2>/dev/null || python3 -c "import time; print(int(time.time()*1000))")

# Only sync C++ source/header files and CMakeLists — nothing else
rsync -avz --relative \
  --include='*/' \
  --include='*.cc' \
  --include='*.h' \
  --include='*.hpp' \
  --include='CMakeLists.txt' \
  --exclude='*' \
  "$REPO_ROOT/cpp/" \
  "$HOST:$REMOTE_DIR/cpp/"

# Also sync any changed Python model files (for compile changes)
rsync -avz --relative \
  --include='*/' \
  --include='*.py' \
  --exclude='*' \
  "$REPO_ROOT/python/mlc_llm/model/dflash/" \
  "$HOST:$REMOTE_DIR/python/mlc_llm/model/dflash/"

SYNC_END=$(date +%s%3N 2>/dev/null || python3 -c "import time; print(int(time.time()*1000))")
echo "Sync took $(( (SYNC_END - START) ))ms"

# ── Step 3: Incremental rebuild on remote ──
echo "=== Rebuilding on $HOST ==="
ssh "$HOST" bash -s "$REMOTE_DIR" <<'REMOTE_BUILD'
  set -euo pipefail
  REMOTE_DIR="$1"
  cd "$REMOTE_DIR"

  export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

  NPROC=$(nproc)
  echo "Building with $NPROC jobs..."
  START_S=$(date +%s%3N)
  cmake --build build --target mlc_llm_module -j"$NPROC" 2>&1
  END_S=$(date +%s%3N)
  echo "Build took $(( (END_S - START_S) ))ms"

  if command -v ccache &>/dev/null; then
    ccache -s 2>/dev/null | grep -E "(Hits|Misses)" || true
  fi
REMOTE_BUILD

TOTAL_END=$(date +%s%3N 2>/dev/null || python3 -c "import time; print(int(time.time()*1000))")
echo "=== Total: $(( (TOTAL_END - START) ))ms ==="

# ── Step 4: Optionally run a test command ──
if [[ -n "$TEST_CMD" ]]; then
  echo ""
  echo "=== Running: $TEST_CMD ==="
  ssh "$HOST" bash -c "
    cd $REMOTE_DIR
    source .venv/bin/activate 2>/dev/null || true
    export PATH=/usr/local/cuda/bin:\$HOME/.cargo/bin:\$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\${LD_LIBRARY_PATH:-}
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
    export PYTHONPATH=\$PWD/3rdparty/tvm/3rdparty/tvm-ffi/python:\$PWD/3rdparty/tvm/python:\$PWD/python:\${PYTHONPATH:-}
    export TVM_LIBRARY_PATH=\$PWD/3rdparty/tvm/build
    export MLC_LLM_LIB_PATH=\$PWD/build
    $TEST_CMD
  "
fi
