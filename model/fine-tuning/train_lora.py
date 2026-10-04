"""LoRA on top of our quantised GGUF (quantise first, then fine-tune).
The HF model's LM linears + tied embedding are replaced by the dequantised GGUF tensors (inverse of gptq_iq.py's HF->GGUF layout), the
base stays frozen, and a LoRA adapter is trained with the frozen prompt on BRACOL dev (val = the 82 dev images with int(sha,16) % 5 == 0)
plus an optional extra manifest (--extra, e.g. the 559 JMuBEN training originals). With --vision-lora the vision blocks start from the
canonical Q8_0 mmproj and get LoRA too. Loss = cross-entropy over the six option letters at the answer position (the harness reads the
same renormalised letter probabilities). Saves DIR/epochN after every epoch and DIR itself at the best val F1.
Inputs: --image-dir holds the 512 px BRACOL dev images by basename (data/prep_images.py writes $WORK/bracol/dev512). Needs one CUDA GPU.
usage: train_lora.py --gguf BASE.gguf --out DIR [--check-only] [--rank 8 --epochs 4 --lr 2e-4 ...]   (release command: README.md)"""
import argparse, csv, json, os, sys, time, types
import numpy as np, torch
import gguf
from gguf import GGUFReader
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

H = os.environ.get("WORK", "work")                                   # data, models and outputs (see env.example.sh at the repository root)
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")   # manifest, prompt, harness
ap = argparse.ArgumentParser()
ap.add_argument("--hf", default=f"{H}/models/Qwen3.5-2B"); ap.add_argument("--gguf", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--manifest", default=f"{REPO}/data/manifest.csv"); ap.add_argument("--image-dir", default=f"{H}/bracol/dev512")
ap.add_argument("--prompt", default=f"{REPO}/protocol/prompt_v1.txt"); ap.add_argument("--ref-rows", default=None)
ap.add_argument("--rank", type=int, default=8); ap.add_argument("--alpha", type=float, default=16); ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--accum", type=int, default=8); ap.add_argument("--check-only", action="store_true"); ap.add_argument("--no-load", action="store_true")
ap.add_argument("--aug", action="store_true", help="random flips / 90-degree rotations of the 512 px training images")
ap.add_argument("--extra", default=None, help="extra training manifest (path,label_letter) with image paths relative to --extra-dir")
ap.add_argument("--extra-dir", default=None); ap.add_argument("--dev-repeat", type=int, default=1); ap.add_argument("--class-weight", action="store_true"); ap.add_argument("--vision-lora", action="store_true"); ap.add_argument("--mmproj", default=f"{H}/gguf/mmproj-Qwen3.5-2B-Q8_0.gguf"); ap.add_argument("--all-dev", action="store_true", help="train on all 419 dev images (val then in-sample)")
ap.add_argument("--targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj_qkv,in_proj_z,out_proj")
a = ap.parse_args(); dev = "cuda:0"; torch.manual_seed(0); import random as _r; _r.seed(0)

model = AutoModelForImageTextToText.from_pretrained(a.hf, dtype=torch.bfloat16).to(dev); proc = AutoProcessor.from_pretrained(a.hf)
def _pe_linear(self, x):                       # same fast patch embed as gptq_iq.py (Conv3d kernel = stride)
    W = self.proj.weight; return torch.nn.functional.linear(x.reshape(-1, W[0].numel()).to(W.dtype), W.reshape(W.shape[0], -1), self.proj.bias)
model.model.visual.patch_embed.forward = types.MethodType(_pe_linear, model.model.visual.patch_embed)
tc = model.config.text_config; NK, NV, HD = tc.linear_num_key_heads, tc.linear_num_value_heads, tc.linear_key_head_dim
perm = torch.arange(NV * HD).reshape(NK, NV // NK, HD).permute(1, 0, 2).reshape(-1); inv = torch.empty_like(perm); inv[perm] = torch.arange(len(perm))

# ---- load the quantised GGUF weights (dequantised) into the HF model, HF layout
T = {t.name: t for t in GGUFReader(a.gguf).tensors} if not a.no_load else {}; t0 = time.time(); nload = 0
def deq(name):
    t = T[name]; x = gguf.quants.dequantize(np.asarray(t.data), t.tensor_type)
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).reshape(int(t.shape[1]) if len(t.shape) > 1 else -1, int(t.shape[0]))
lm = model.model.language_model
with torch.no_grad():
  if not a.no_load:
    for li, layer in enumerate(lm.layers):
        if hasattr(layer, "linear_attn"):
            la = layer.linear_attn
            W = deq(f"blk.{li}.attn_qkv.weight"); W = torch.cat([W[:NK*HD], W[NK*HD:2*NK*HD], W[2*NK*HD:][inv]], 0); la.in_proj_qkv.weight.copy_(W)
            la.in_proj_z.weight.copy_(deq(f"blk.{li}.attn_gate.weight")[inv]); la.out_proj.weight.copy_(deq(f"blk.{li}.ssm_out.weight")[:, inv]); nload += 3
        else:
            sa = layer.self_attn
            for g, m in (("attn_q", sa.q_proj), ("attn_k", sa.k_proj), ("attn_v", sa.v_proj), ("attn_output", sa.o_proj)): m.weight.copy_(deq(f"blk.{li}.{g}.weight")); nload += 1
        for g, m in (("ffn_gate", layer.mlp.gate_proj), ("ffn_up", layer.mlp.up_proj), ("ffn_down", layer.mlp.down_proj)): m.weight.copy_(deq(f"blk.{li}.{g}.weight")); nload += 1
    E = deq("token_embd.weight"); lm.embed_tokens.weight.copy_(E); nload += 1
    assert model.lm_head.weight.data_ptr() == lm.embed_tokens.weight.data_ptr() or torch.equal(model.lm_head.weight, lm.embed_tokens.weight)
print(f"loaded {nload} dequantised tensors from {os.path.basename(a.gguf)} in {time.time()-t0:.0f}s", flush=True)
VMAP = {"attn.qkv": "attn_qkv", "attn.proj": "attn_out", "mlp.linear_fc1": "ffn_up", "mlp.linear_fc2": "ffn_down"}
if a.vision_lora:                               # deployed vision weights = canonical Q8_0 mmproj, dequantised (same names 1:1)
    VT = {t.name: t for t in GGUFReader(a.mmproj).tensors}; vis = model.model.visual; worst = 0.0
    def vdeq(n):
        t = VT[n]; x = gguf.quants.dequantize(np.asarray(t.data), t.tensor_type); return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).reshape(int(t.shape[1]), int(t.shape[0]))
    with torch.no_grad():
        for i, blk in enumerate(vis.blocks):
            for hn, gn in VMAP.items():
                mod = blk.get_submodule(hn); W = vdeq(f"v.blk.{i}.{gn}.weight"); worst = max(worst, float((W - mod.weight.float().cpu()).abs().max() / mod.weight.float().abs().max())); mod.weight.copy_(W)
        for hn, gn in (("merger.linear_fc1", "mm.0"), ("merger.linear_fc2", "mm.2")):
            mod = vis.get_submodule(hn); W = vdeq(f"{gn}.weight"); worst = max(worst, float((W - mod.weight.float().cpu()).abs().max() / mod.weight.float().abs().max())); mod.weight.copy_(W)
    print(f"vision: Q8_0 weights loaded from {os.path.basename(a.mmproj)}, worst rel. diff vs bf16 {worst:.4f}", flush=True); assert worst < 0.05

txt = open(a.prompt, encoding="utf-8").read()
sys_t = txt.split("### system\n")[1].split("### user\n")[0].strip(); usr_t = txt.split("### user\n")[1].split("### grammar")[0].strip()
L = "ABCDEF"; lid = [proc.tokenizer.convert_tokens_to_ids(c) for c in L]; assert all(isinstance(i, int) and i >= 0 for i in lid), lid
rows = [r for r in csv.DictReader(open(a.manifest)) if r["split"] == "dev"]
val = [r for r in rows if int(r["sha256"], 16) % 5 == 0]; tr = [r for r in rows if int(r["sha256"], 16) % 5 != 0] if not a.all_dev else list(rows)
tr = tr * a.dev_repeat
if a.extra:
    ex = [dict(path=os.path.join(a.extra_dir, r["path"]), label_letter=r["label_letter"], sha256=r.get("sha256", ""), split="extra") for r in csv.DictReader(open(a.extra)) if r["label_letter"] in "ABCDE"]
    tr = tr + ex; print(f"extra training images: {len(ex)} from {a.extra}", flush=True)
print(f"dev {len(rows)}: train {len(tr)}, val {len(val)}", flush=True)
cache = {}
import random
def load_img(r):
    return Image.open(r["path"] if r.get("split") == "extra" else os.path.join(a.image_dir, os.path.basename(r["path"]))).convert("RGB")
def inputs(r, train=False):
    if train and a.aug:
        im = load_img(r); k = random.randrange(8)
        if k & 1: im = im.transpose(Image.FLIP_LEFT_RIGHT)
        im = im.rotate(90 * (k >> 1), expand=True)
        msgs = [{"role": "system", "content": [{"type": "text", "text": sys_t}]}, {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": usr_t}]}]
        chat = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return {k2: v.to(dev) for k2, v in proc(text=[chat], images=[im], return_tensors="pt").items()}
    if r["path"] not in cache:
        im = load_img(r)
        msgs = [{"role": "system", "content": [{"type": "text", "text": sys_t}]}, {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": usr_t}]}]
        chat = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        cache[r["path"]] = proc(text=[chat], images=[im], return_tensors="pt")
    return {k: v.to(dev) for k, v in cache[r["path"]].items()}
def letter_logits(r, train=False):
    out = model(**inputs(r, train), use_cache=False); return out.logits[0, -1, lid].float()
def evaluate(R):
    model.eval(); P = []
    with torch.no_grad():
        for r in R: P.append(torch.softmax(letter_logits(r), -1).cpu().numpy())
    P = np.array(P); y = [r["label_letter"] for r in R]; pred = [L[i] for i in P.argmax(1)]
    f1 = []
    for c in sorted(set(y)):
        tp = sum(p == c and t == c for p, t in zip(pred, y)); fp = sum(p == c and t != c for p, t in zip(pred, y)); fn = sum(p != c and t == c for p, t in zip(pred, y))
        f1.append(0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1)), float(np.mean([p == t for p, t in zip(pred, y)])), P

if a.ref_rows:                                  # check the load against the llama-server rows of the same GGUF
    ref = {r["path"]: r for r in csv.DictReader(open(a.ref_rows))}; R = [r for r in rows[:8]]
    _, _, P = evaluate(R)
    for r, p in zip(R, P):
        q = np.array([float(ref[r["path"]][f"p_{c}"]) for c in L]); print(f"  {r['path']}: hf {np.round(p,3)} server {np.round(q,3)} |d|max {np.abs(p-q).max():.3f}", flush=True)
f1v, accv, _ = evaluate(val); print(f"base (no adapter) val F1 {f1v:.4f} acc {accv:.4f}", flush=True)
if a.check_only: sys.exit(0)

from peft import LoraConfig, get_peft_model
tg = "|".join(a.targets.split(","))
tre = rf".*language_model\.layers\.\d+\..*\.({tg})$" + (r"|.*visual\.(blocks\.\d+\.(attn\.qkv|attn\.proj|mlp\.linear_fc1|mlp\.linear_fc2)|merger\.linear_fc1|merger\.linear_fc2)$" if a.vision_lora else "")
cfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05, target_modules=tre, bias="none")
model = get_peft_model(model, cfg); model.print_trainable_parameters()
from collections import Counter as _C
_cnt = _C(r['label_letter'] for r in tr); cw = torch.tensor([(len(tr) / (5 * _cnt[c])) ** 0.5 if _cnt[c] else 0.0 for c in L], device=dev) if a.class_weight else None
if cw is not None: print('class weights', [round(float(x), 2) for x in cw], flush=True)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
steps = a.epochs * len(tr) // a.accum; sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / max(1, steps)))
best = (f1v, -1); hist = []
for ep in range(a.epochs):
    model.train(); g = torch.Generator().manual_seed(ep); order = torch.randperm(len(tr), generator=g).tolist(); tl = 0.0; t0 = time.time()
    for i, k in enumerate(order):
        r = tr[k]; y = torch.tensor(L.index(r["label_letter"]), device=dev)
        loss = torch.nn.functional.cross_entropy(letter_logits(r, True)[None], y[None], weight=cw) / a.accum; loss.backward(); tl += loss.item() * a.accum
        if (i + 1) % a.accum == 0: opt.step(); sched.step(); opt.zero_grad()
    f1v, accv, _ = evaluate(val); hist.append(dict(epoch=ep + 1, train_loss=tl / len(tr), val_f1=f1v, val_acc=accv))
    print(f"epoch {ep+1}: train loss {tl/len(tr):.4f} val F1 {f1v:.4f} acc {accv:.4f} ({time.time()-t0:.0f}s)", flush=True)
    if f1v > best[0] or a.all_dev: best = (f1v, ep + 1); model.save_pretrained(a.out)
    model.save_pretrained(os.path.join(a.out, f"epoch{ep+1}"))
json.dump(dict(base=a.gguf, rank=a.rank, alpha=a.alpha, lr=a.lr, epochs=a.epochs, targets=a.targets, train=len(tr), val=len(val), history=hist,
               best_epoch=best[1], best_val_f1=best[0]), open(os.path.join(a.out, "train_log.json") if os.path.isdir(a.out) else a.out + ".log.json", "w"), indent=1)
print(f"best val F1 {best[0]:.4f} at epoch {best[1]} -> {a.out}", flush=True)
