"""Helper script to compile DFlash draft model with full TVM (Metal+LLVM / CUDA+LLVM support)."""
import sys
import platform
# Load the full TVM compiler library first
from tvm_ffi.module import load_module as _load_tvm_module
_ext = ".dylib" if platform.system() == "Darwin" else ".so"
_load_tvm_module(
    f"3rdparty/tvm/build/libtvm{_ext}",
    keep_module_alive=True,
)
# Now run the MLC LLM compile CLI
from mlc_llm.cli.compile import main
main(sys.argv[1:])
