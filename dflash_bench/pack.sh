#!/usr/bin/env bash
# Create a tarball of the repo for upload to remote GPU instances.
# This bypasses SkyPilot's .gitignore-based rsync which excludes
# critical source files (target/, debug/, tvm_ffi python package, etc.).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TARBALL="$SCRIPT_DIR/repo.tar.gz"

echo "Creating tarball of $REPO_DIR..."
# COPYFILE_DISABLE prevents macOS from including ._* AppleDouble resource fork files
COPYFILE_DISABLE=1 tar czf "$TARBALL" \
  -C "$REPO_DIR" \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='build' \
  --exclude='dist' \
  --exclude='hf_models' \
  --exclude='__pycache__' \
  --exclude='*.o' \
  --exclude='*.so' \
  --exclude='*.dylib' \
  --exclude='*.a' \
  --exclude='*.lib' \
  --exclude='.DS_Store' \
  --exclude='dflash_bench/repo.tar.gz' \
  --exclude='dflash_bench/results' \
  --exclude='._*' \
  .

echo "Tarball created: $TARBALL ($(du -h "$TARBALL" | cut -f1))"
echo ""
echo "Next: sky launch dflash_bench/sky_bench.yaml -c dflash-bench -y"
