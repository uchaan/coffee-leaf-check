"""Template plan: start from Unsloth file BASE's types, set token_embd to EMB, then move tensors toward Unsloth file UP's types (in a fixed
priority order: full-attention q/k/v/o, ffn_down, ffn_gate, ffn_up, linear-attention qkv/gate/out) while the file stays <= BUDGET, or,
if over budget, toward Unsloth file DOWN's types in the reverse order until it fits. usage: plan_alloc.py BASE EMB BUDGET UP|- DOWN|- OUT"""
import sys, os; sys.path.insert(0, os.path.join(os.environ.get("LLAMA_CPP_DIR", "llama.cpp"), "gguf-py"))
from gguf import GGUFReader
import gguf
base, emb, budget, up, down, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
B = {t.name: t for t in GGUFReader(base).tensors}
def nbytes(t, typ):
    bs, ts = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType[typ]]; return int(t.n_elements) // bs * ts
size = os.path.getsize(base) - int(B["token_embd.weight"].n_bytes) + nbytes(B["token_embd.weight"], emb); chg = {"token_embd.weight": emb}
order = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_down", "ffn_gate", "ffn_up", "attn_qkv", "attn_gate", "ssm_out"]
def prio(n): k = n.split(".")[2]; return (order.index(k) if k in order else 99, int(n.split(".")[1]))
cap = budget - 300_000                                                  # GGUF alignment margin
if size > cap and down != "-":
    D = {t.name: t for t in GGUFReader(down).tensors}
    for n in sorted([n for n in B if n.startswith("blk.") and n in D and int(D[n].n_bytes) < int(B[n].n_bytes)], key=prio, reverse=True):
        if size <= cap: break
        size -= int(B[n].n_bytes) - int(D[n].n_bytes); chg[n] = D[n].tensor_type.name
elif size < cap and up != "-":
    U = {t.name: t for t in GGUFReader(up).tensors}
    for n in sorted([n for n in B if n.startswith("blk.") and n in U and int(U[n].n_bytes) > int(B[n].n_bytes)], key=prio):
        d = int(U[n].n_bytes) - int(B[n].n_bytes)
        if size + d <= cap: size += d; chg[n] = U[n].tensor_type.name
print(f"{out}: planned ~{size:,} B (budget {budget:,}); {len(chg)-1} body tensors changed", file=sys.stderr)
open(out, "w").write("".join(f"{k}={v}\n" for k, v in chg.items()))
