# ctypes binding to libggml: quantize float rows (optionally with per-column imatrix weights) into any ggml type.
import ctypes, os, numpy as np, gguf
_lib=ctypes.CDLL(os.path.join(os.environ.get("LLAMA_CPP_DIR", "llama.cpp"), "build", "bin", "libggml-base.so"))
_lib.ggml_quantize_chunk.restype=ctypes.c_size_t
_lib.ggml_quantize_chunk.argtypes=[ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_float)]
_lib.ggml_type_size.restype=ctypes.c_size_t; _lib.ggml_blck_size.restype=ctypes.c_int64
def quantize(x: np.ndarray, qtype, imatrix=None) -> np.ndarray:
    """x: float32 [nrows, n_per_row]; imatrix: float32 [n_per_row] or None. Returns uint8 [nrows, row_bytes]."""
    x=np.ascontiguousarray(x, dtype=np.float32); nrows, n=x.shape
    ts=_lib.ggml_type_size(int(qtype)); bs=_lib.ggml_blck_size(int(qtype)); row_bytes=n//bs*ts
    out=np.zeros((nrows,row_bytes),dtype=np.uint8)
    im=None if imatrix is None else np.ascontiguousarray(imatrix,dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    got=_lib.ggml_quantize_chunk(int(qtype), x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out.ctypes.data_as(ctypes.c_void_p), 0, nrows, n, im)
    assert got==out.nbytes, (got, out.nbytes); return out
