"""bytes, bpw (llama.cpp's definition: sum of tensor bytes * 8 / sum of tensor elements), tensor count and sha256 of a GGUF."""
import sys, os, hashlib
sys.path.insert(0, os.path.join(os.environ.get("LLAMA_CPP_DIR", "llama.cpp"), "gguf-py"))
from gguf import GGUFReader
for p in sys.argv[1:]:
    r = GGUFReader(p); nb = sum(int(t.n_bytes) for t in r.tensors); ne = sum(int(t.n_elements) for t in r.tensors)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""): h.update(chunk)
    print(f"{os.path.basename(p)},{os.path.getsize(p)},{8*nb/ne:.4f},{len(r.tensors)},{h.hexdigest()}")
