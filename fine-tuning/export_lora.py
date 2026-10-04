"""Export a train_lora.py adapter for llama.cpp: LM LoRA tensors -> a PEFT dir for llama.cpp's convert_lora_to_gguf.py (Q8_0 GGUF adapter);
vision LoRA tensors (if any) are merged into the canonical Q8_0 mmproj (W = dequant(Q8_0) + scale * B @ A, re-quantised to Q8_0) and a new
mmproj GGUF is written with every other tensor and key copied unchanged. Uses gguf-py and convert_lora_to_gguf.py from $LLAMA_CPP_DIR
(the pinned llama.cpp commit), not the pip gguf package.
usage: export_lora.py ADAPTER_DIR OUT_PREFIX [--mmproj CANONICAL] [--hf HF_DIR]   -> OUT_PREFIX-lm-q8.gguf [, OUT_PREFIX-mmproj-Q8_0.gguf]"""
import argparse, json, os, subprocess, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.environ.get("LLAMA_CPP_DIR", "llama.cpp"), "gguf-py"))
import gguf
from gguf import GGUFReader, GGUFWriter
from safetensors.torch import load_file, save_file

H = os.environ.get("WORK", "work")                                   # data, models and outputs (see ../agentic-quantization/loop/env.example.sh)
LL = os.environ.get("LLAMA_CPP_DIR", "llama.cpp")
ap = argparse.ArgumentParser(); ap.add_argument("adapter"); ap.add_argument("out"); ap.add_argument("--mmproj", default=f"{H}/gguf/mmproj-Qwen3.5-2B-Q8_0.gguf")
ap.add_argument("--hf", default=f"{H}/models/Qwen3.5-2B", help="HF base model dir (same default as train_lora.py)")
a = ap.parse_args()
cfg = json.load(open(os.path.join(a.adapter, "adapter_config.json"))); sd = load_file(os.path.join(a.adapter, "adapter_model.safetensors"))
scale = cfg["lora_alpha"] / cfg["r"]
lm = {k: v for k, v in sd.items() if ".visual." not in k}; vis = {k: v for k, v in sd.items() if ".visual." in k}
d = a.out + "-lm"; os.makedirs(d, exist_ok=True); save_file(lm, os.path.join(d, "adapter_model.safetensors"))
c2 = dict(cfg); c2["target_modules"] = cfg["target_modules"].split("|.*visual")[0]; json.dump(c2, open(os.path.join(d, "adapter_config.json"), "w"), indent=1)
subprocess.run([sys.executable, f"{LL}/convert_lora_to_gguf.py", "--base", a.hf, "--outfile", a.out + "-lm-q8.gguf", "--outtype", "q8_0", d],
               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(f"LM adapter: {len(lm)//2} pairs -> {a.out}-lm-q8.gguf ({os.path.getsize(a.out + '-lm-q8.gguf'):,} B)")
if not vis: sys.exit(0)

VMAP = {"attn.qkv": "attn_qkv", "attn.proj": "attn_out", "mlp.linear_fc1": "ffn_up", "mlp.linear_fc2": "ffn_down"}
def gname(mod):                                  # HF visual module path -> mmproj tensor name
    if mod.startswith("merger.linear_fc1"): return "mm.0.weight"
    if mod.startswith("merger.linear_fc2"): return "mm.2.weight"
    i = int(mod.split(".")[1]); rest = ".".join(mod.split(".")[2:]); return f"v.blk.{i}.{VMAP[rest]}.weight"
deltas = {}
for k in vis:
    if not k.endswith("lora_A.weight"): continue
    mod = k.split(".visual.")[1].rsplit(".lora_A", 1)[0]; A = vis[k].float(); B = vis[k.replace("lora_A", "lora_B")].float()
    deltas[gname(mod)] = (B @ A * scale).numpy()
R = GGUFReader(a.mmproj); arch = bytes(R.fields["general.architecture"].parts[-1]).decode()
out = a.out + "-mmproj-Q8_0.gguf"; w = GGUFWriter(out, arch=arch, endianess=R.endianess)
for f in R.fields.values():
    if f.name == "general.architecture" or f.name.startswith("GGUF."): continue
    vt = f.types[0]; st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None; w.add_key_value(f.name, f.contents(), vt, sub_type=st)
data = {}; nchg = 0
for t in R.tensors:
    if t.name in deltas:
        W = gguf.quants.dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32).reshape(int(t.shape[1]), int(t.shape[0]))
        Wn = W + deltas[t.name]; q = gguf.quants.quantize(Wn, t.tensor_type); data[t.name] = (q, t.tensor_type); nchg += 1
        w.add_tensor_info(t.name, q.shape, q.dtype, q.nbytes, t.tensor_type)
    else:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for t in R.tensors: w.write_tensor_data(data[t.name][0] if t.name in data else np.asarray(t.data), tensor_endianess=R.endianess)
w.close(); print(f"mmproj: {nchg} tensors merged and re-quantised -> {out} ({os.path.getsize(out):,} B; canonical {os.path.getsize(a.mmproj):,} B)")
