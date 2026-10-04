#!/usr/bin/env python
"""GPTQ-IQ: sequential Hessian-aware quantisation of Qwen3.5 (hybrid linear/full attention) models into llama.cpp IQ/K formats,
with image+text calibration for the VLMs and GPTQ of the tied embedding as the output head.

Per linear: H = X^T X from calibration activations of the (already-quantised) upstream model; 256-column blocks are
rounded with llama.cpp's own quantiser (libggml, imatrix = diag H) and the joint block error is propagated to the
remaining columns with the exact block-OBS update  W_R -= (W_B - Q_B) U_BB^{-1} U_BR,  U = chol(H^{-1}) upper.
Output: the template GGUF with quantised tensors' bytes replaced (types, order, bytes identical)."""
import argparse, json, math, os, sys, time
import numpy as np, torch, gguf
from concurrent.futures import ThreadPoolExecutor
from gguf import GGUFReader, GGUFWriter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ggml_quant

NK, NV, HD = 16, 16, 128          # linear_num_key_heads, linear_num_value_heads, head dim; overwritten from the model config in main()
def v_perm(head_dim):              # GGUF row r <- HF row perm[r]  (converter _reorder_v_heads, grouped -> tiled)
    return torch.arange(NV*head_dim).reshape(NK, NV//NK, head_dim).permute(1, 0, 2).reshape(-1)

def ggml_quantize_threads(x: np.ndarray, qtype, imat: np.ndarray, threads=16) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32); rows, n = x.shape
    ts = ggml_quant._lib.ggml_type_size(int(qtype)); bs = ggml_quant._lib.ggml_blck_size(int(qtype)); rb = n // bs * ts
    out = np.zeros((rows, rb), dtype=np.uint8); im = np.ascontiguousarray(imat, dtype=np.float32)
    import ctypes
    def work(r0, r1):
        got = ggml_quant._lib.ggml_quantize_chunk(int(qtype), x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out.ctypes.data_as(ctypes.c_void_p),
                                                  r0 * n, r1 - r0, n, im.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        assert got == (r1 - r0) * rb, (got, r1 - r0, rb)
    step = max(1, math.ceil(rows / threads)); bounds = [(r, min(rows, r + step)) for r in range(0, rows, step)]
    with ThreadPoolExecutor(threads) as ex: list(ex.map(lambda b: work(*b), bounds))
    return out

def dequant(qbytes: np.ndarray, qtype) -> np.ndarray:
    return gguf.quants.dequantize(qbytes, qtype).astype(np.float32)

@torch.no_grad()
def gptq_iq(W: torch.Tensor, H: torch.Tensor, qtype, percdamp=0.01, log=None, block_weights="diag", block_order="seq"):
    """W [rows, cols] fp32 (GGUF layout), H [cols, cols] fp32 sum X^T X. Returns (qbytes uint8 [rows,row_bytes], Wq fp32)."""
    dev = W.device; rows, cols = W.shape; assert cols % 256 == 0, cols
    H = H.clone(); W = W.clone()
    dead = torch.diag(H) == 0; H[dead, dead] = 1.0; W[:, dead] = 0.0
    imat = torch.diag(H).clone()                              # imatrix weights = column energy (relative scale only)
    damp = percdamp * torch.mean(torch.diag(H)); H += torch.eye(cols, device=dev) * damp
    L = torch.linalg.cholesky(H); Hinv = torch.cholesky_inverse(L); U = torch.linalg.cholesky(Hinv, upper=True); del L, Hinv
    nb = cols // 256
    if block_order == "act":                                   # block-level act-order: eliminate the most energetic blocks first
        energy = imat.view(nb, 256).sum(1); order = torch.argsort(energy, descending=True).tolist()
        perm = torch.cat([torch.arange(b*256, (b+1)*256, device=dev) for b in order]); inv = torch.empty_like(perm); inv[perm] = torch.arange(cols, device=dev)
        W = W[:, perm]; H = H[perm][:, perm]; imat = imat[perm]
        L = torch.linalg.cholesky(H); Hinv = torch.cholesky_inverse(L); U = torch.linalg.cholesky(Hinv, upper=True); del L, Hinv
    else: perm = inv = None
    blocks = []; Wq = torch.empty_like(W); err_h = 0.0
    for i0 in range(0, cols, 256):
        i1 = i0 + 256; Wb = W[:, i0:i1]
        if block_weights == "cond":                            # diag of (Hinv_cond)_BB^-1 = U_BB^-1 U_BB^-T: conditional importance
            Uinv = torch.linalg.solve_triangular(U[i0:i1, i0:i1], torch.eye(256, device=dev), upper=True); wcol = (Uinv * Uinv).sum(1)
        else: wcol = imat[i0:i1]
        qb = ggml_quantize_threads(Wb.cpu().numpy(), qtype, wcol.cpu().numpy())
        Qb = torch.from_numpy(dequant(qb, qtype)).to(dev); Wq[:, i0:i1] = Qb; blocks.append(qb)
        E = Wb - Qb                                            # joint block error
        if i1 < cols:
            Ubb = U[i0:i1, i0:i1]; Err = torch.linalg.solve_triangular(Ubb.T, E.T, upper=False).T   # E U_BB^{-1}
            W[:, i1:] -= Err @ U[i0:i1, i1:]
    if perm is not None:                                       # undo the block permutation: bytes back to natural block order, Wq to natural columns
        blocks = [blocks[order.index(b)] for b in range(nb)]; Wq = Wq[:, inv]
    return np.concatenate(blocks, axis=1), Wq

def h_weighted_err(W, Q, H):                                   # tr((W-Q) H (W-Q)^T): layer-output error proxy
    D = (W - Q); return float((D @ H * D).sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True); ap.add_argument("--template", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--calib", nargs="+", required=True); ap.add_argument("--nseq", type=int, default=0); ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=-1, help="quantise only the first N layers (smoke test)")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--mb", type=int, default=1); ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--percdamp", type=float, default=0.01); ap.add_argument("--skip-lm-head", action="store_true")
    ap.add_argument("--tied-embd-gptq", action="store_true", help="tied embeddings: GPTQ token_embd as the output head (H from final hidden states)")
    ap.add_argument("--calib-images", default=None, help="manifest.csv: its split=dev rows (only) are added as image+prompt calibration samples")
    ap.add_argument("--image-dir", default=None, help="folder with the preprocessed (512 px, JPEG q90) dev images, by basename")
    ap.add_argument("--prompt-file", default=None, help="protocol prompt file (### system / ### user / ### grammar)")
    ap.add_argument("--block-weights", default="diag", choices=["diag","cond"]); ap.add_argument("--block-order", default="seq", choices=["seq","act"]); ap.add_argument("--offload-layers", action="store_true", help="decoder on CPU, one layer on the GPU at a time (low GPU memory)")
    a = ap.parse_args(); dev = torch.device(a.device); t_start = time.time(); torch.set_grad_enabled(False)
    print(f"settings: mb={a.mb}, lm_head offload" + (", layer offload" if a.offload_layers else ""), flush=True)
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(a.hf)
    ids = []
    for f in a.calib: ids += tok(open(f, encoding="utf-8", errors="ignore").read(), add_special_tokens=False)["input_ids"]
    n = len(ids) // a.seqlen; data = torch.tensor(ids[: n * a.seqlen]).view(n, a.seqlen)
    if a.nseq: data = data[: a.nseq]
    print(f"calibration: {data.shape[0]} x {a.seqlen} tokens from {a.calib}", flush=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(a.hf, dtype=torch.bfloat16, device_map={"": a.device}); model.eval()
    global NK, NV, HD                                             # linear-attention head layout from the config
    tc = model.config.text_config; NK, NV, HD = tc.linear_num_key_heads, tc.linear_num_value_heads, tc.linear_key_head_dim
    assert tc.linear_value_head_dim == HD, (tc.linear_key_head_dim, tc.linear_value_head_dim)
    print(f"linear attention heads: key {NK}, value {NV}, head dim {HD}", flush=True)
    lm = model.model.language_model; layers = lm.layers
    T = GGUFReader(a.template); ttype = {t.name: t.tensor_type for t in T.tensors}
    QTYPES = {n for n, ty in ttype.items() if ty.name not in ("F32", "F16", "BF16")}
    # --- layer-0 inputs and the kwargs the model passes to layers (captured once) ---
    cap = {}
    class Catcher(torch.nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, hs, **kw): cap["kw"] = kw; cap["hs"] = hs; raise StopIteration
    layers[0] = Catcher(layers[0])
    try: model(input_ids=data[: a.mb].to(dev))
    except StopIteration: pass
    kw = cap["kw"]; pe_full = kw["position_embeddings"]; pid = kw.get("position_ids")
    img = []                                                       # (hs [1,L,H], position_embeddings, position_ids) per dev image
    if a.calib_images:
        import csv; from PIL import Image; from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(a.hf); txt = open(a.prompt_file, encoding="utf-8").read()
        import types                                               # HF Conv3d patch embed falls back to slow_conv_dilated3d (3.5 s/img):
        def _pe_linear(self, x):                                   # kernel = stride = patch, so it is one matmul (rel. diff 5e-5, verified)
            W = self.proj.weight; return torch.nn.functional.linear(x.reshape(-1, W[0].numel()).to(W.dtype), W.reshape(W.shape[0], -1), self.proj.bias)
        model.model.visual.patch_embed.forward = types.MethodType(_pe_linear, model.model.visual.patch_embed)
        sys_t = txt.split("### system\n")[1].split("### user\n")[0].strip(); usr_t = txt.split("### user\n")[1].split("### grammar")[0].strip()
        rows = list(csv.DictReader(open(a.calib_images))); dev_rows = [r for r in rows if r["split"] == "dev"]
        assert all(r["split"] in ("dev", "test") for r in rows) and len(dev_rows) > 0
        for r in dev_rows:
            im = Image.open(os.path.join(a.image_dir, os.path.basename(r["path"]))).convert("RGB")
            msgs = [{"role": "system", "content": [{"type": "text", "text": sys_t}]}, {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": usr_t}]}]
            chat = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inp = {k: v.to(dev) for k, v in proc(text=[chat], images=[im], return_tensors="pt").items()}
            try: model(**inp, use_cache=False)
            except StopIteration: pass
            img.append((cap["hs"].detach().to(torch.bfloat16), cap["kw"]["position_embeddings"], cap["kw"].get("position_ids")))
        print(f"image calibration: {len(img)} dev images (split=dev only), {sum(h.shape[1] for h, _, _ in img)} tokens, prompt {a.prompt_file}", flush=True)
    layers[0] = layers[0].m
    if a.offload_layers: lm.layers.to("cpu"); torch.cuda.empty_cache()
    hs = lm.embed_tokens(data.to(dev)).to(torch.bfloat16)          # [n, L, hidden]
    def layer_kwargs(b):
        pe = tuple(p[..., :b, :, :] if p.dim() == 4 else p[:b] for p in pe_full) if isinstance(pe_full, tuple) else pe_full
        return dict(position_embeddings=pe, attention_mask=None, position_ids=(pid[:b] if pid is not None else None))
    def run_layer(layer, hs):
        out = torch.empty_like(hs)
        for i in range(0, hs.shape[0], a.mb):
            b = min(a.mb, hs.shape[0] - i); out[i:i+b] = layer(hs[i:i+b], **layer_kwargs(b))
        return out
    results = {}; stats = []
    nl = len(layers) if a.layers < 0 else a.layers
    for li in range(nl):
        layer = layers[li]; t0 = time.time()
        if a.offload_layers: layer.to(dev)
        lins = {}                                               # gguf name -> (module, hf->gguf transform kind)
        if hasattr(layer, "linear_attn"):
            la = layer.linear_attn
            lins[f"blk.{li}.attn_qkv.weight"] = (la.in_proj_qkv, "qkv"); lins[f"blk.{li}.attn_gate.weight"] = (la.in_proj_z, "z")
            lins[f"blk.{li}.ssm_out.weight"] = (la.out_proj, "out")
        else:
            sa = layer.self_attn
            for g, m in (("attn_q", sa.q_proj), ("attn_k", sa.k_proj), ("attn_v", sa.v_proj), ("attn_output", sa.o_proj)): lins[f"blk.{li}.{g}.weight"] = (m, "id")
        for g, m in (("ffn_gate", layer.mlp.gate_proj), ("ffn_up", layer.mlp.up_proj), ("ffn_down", layer.mlp.down_proj)): lins[f"blk.{li}.{g}.weight"] = (m, "id")
        lins = {k: v for k, v in lins.items() if k in QTYPES}
        # pass 1: collect H per linear
        Hs = {k: torch.zeros(m.in_features, m.in_features, device=dev, dtype=torch.float32) for k, (m, _) in lins.items()}
        hooks = []
        for k, (m, _) in lins.items():
            def mk(k):
                def hook(mod, inp): x = inp[0].reshape(-1, inp[0].shape[-1]).float(); Hs[k].addmm_(x.T, x)
                return hook
            hooks.append(m.register_forward_pre_hook(mk(k)))
        run_layer(layer, hs)
        for h_i, pe_i, pid_i in img: layer(h_i, position_embeddings=pe_i, attention_mask=None, position_ids=pid_i)
        for h in hooks: h.remove()
        # quantise each linear in GGUF layout
        for k, (m, kind) in lins.items():
            qtype = ttype[k]; Whf = m.weight.data.float(); H = Hs[k]
            if kind == "qkv":
                q, kk, v = Whf[:NK*HD], Whf[NK*HD:2*NK*HD], Whf[2*NK*HD:]; perm = v_perm(HD).to(dev); Wg = torch.cat([q, kk, v[perm]], 0)
            elif kind == "z": perm = v_perm(HD).to(dev); Wg = Whf[perm]
            elif kind == "out": perm = v_perm(HD).to(dev); Wg = Whf[:, perm]; H = H[perm][:, perm]
            else: Wg = Whf
            e_stock = None
            qb, Wq = gptq_iq(Wg, H, qtype, a.percdamp, block_weights=a.block_weights, block_order=a.block_order)
            # diagnostics: H-weighted error of stock rounding (same imatrix) vs GPTQ result
            imat = torch.diag(H).clone(); imat[imat == 0] = 1
            qs = ggml_quantize_threads(Wg.cpu().numpy(), qtype, imat.cpu().numpy(), a.threads); Qs = torch.from_numpy(dequant(qs, qtype)).to(dev)
            e_stock = h_weighted_err(Wg, Qs, H); e_gptq = h_weighted_err(Wg, Wq, H)
            results[k] = qb
            # write quantised weights back into the model (HF layout) for the sequential pass
            if kind == "qkv":
                inv = torch.empty_like(perm); inv[perm] = torch.arange(len(perm), device=dev)
                Wback = torch.cat([Wq[:NK*HD], Wq[NK*HD:2*NK*HD], Wq[2*NK*HD:][inv]], 0)
            elif kind == "z": inv = torch.empty_like(perm); inv[perm] = torch.arange(len(perm), device=dev); Wback = Wq[inv]
            elif kind == "out": inv = torch.empty_like(perm); inv[perm] = torch.arange(len(perm), device=dev); Wback = Wq[:, inv]
            else: Wback = Wq
            m.weight.data.copy_(Wback.to(torch.bfloat16))
            stats.append(dict(name=k, type=qtype.name, e_stock=e_stock, e_gptq=e_gptq, gain=1 - e_gptq / max(e_stock, 1e-30)))
            print(f"  {k:28s} {qtype.name:8s} H-err stock {e_stock:.4g} -> gptq {e_gptq:.4g} ({100*(1-e_gptq/max(e_stock,1e-30)):+.1f}%)", flush=True)
            del H, Wg, Wq, Qs
        del Hs; torch.cuda.empty_cache()
        hs = run_layer(layer, hs)                               # pass 2: outputs with quantised weights
        img = [(layer(h_i, position_embeddings=pe_i, attention_mask=None, position_ids=pid_i), pe_i, pid_i) for h_i, pe_i, pid_i in img]
        if a.offload_layers: layer.to("cpu"); torch.cuda.empty_cache()
        print(f"layer {li} done in {time.time()-t0:.0f}s ({len(lins)} tensors) total {time.time()-t_start:.0f}s", flush=True)
    lm.layers.to("cpu"); torch.cuda.empty_cache()            # decoder finished: free it for the lm_head Hessian
    if not a.skip_lm_head and a.layers < 0 and "output.weight" in QTYPES:
        hn = lm.norm(hs); x = hn.reshape(-1, hn.shape[-1]).float(); H = x.T @ x; del x
        W = model.lm_head.weight.data.float(); qb, Wq = gptq_iq(W, H, ttype["output.weight"], a.percdamp, block_weights=a.block_weights, block_order=a.block_order); results["output.weight"] = qb
        print("lm_head done", flush=True)
    elif a.tied_embd_gptq and a.layers < 0 and "token_embd.weight" in QTYPES and "output.weight" not in ttype:
        hn = lm.norm(hs); x = hn.reshape(-1, hn.shape[-1]).float(); H = x.T @ x; del x       # tied head = token_embd
        for h_i, _, _ in img: xi = lm.norm(h_i).reshape(-1, h_i.shape[-1]).float(); H += xi.T @ xi
        W = model.lm_head.weight.data.float(); qb, Wq = gptq_iq(W, H, ttype["token_embd.weight"], a.percdamp, block_weights=a.block_weights, block_order=a.block_order)
        results["token_embd.weight"] = qb; print(f"tied token_embd ({ttype['token_embd.weight'].name}) GPTQ as the output head done", flush=True)
    # --- write GGUF: template with replaced bytes ---
    arch = bytes(T.fields["general.architecture"].parts[-1]).decode(); w = GGUFWriter(a.out, arch=arch, endianess=T.endianess)
    for f in T.fields.values():
        if f.name == "general.architecture" or f.name.startswith("GGUF."): continue
        vt = f.types[0]; st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None; w.add_key_value(f.name, f.contents(), vt, sub_type=st)
    for t in T.tensors: w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    nrep = 0
    for t in T.tensors:
        if t.name in results:
            q = results[t.name]; assert q.nbytes == t.data.nbytes and q.shape == t.data.shape, (t.name, q.shape, t.data.shape)
            w.write_tensor_data(q, tensor_endianess=T.endianess); nrep += 1
        else: w.write_tensor_data(np.asarray(t.data), tensor_endianess=T.endianess)
    w.close()
    json.dump(stats, open(a.out + ".gptq-stats.json", "w"), indent=1)
    print(f"wrote {a.out}: replaced {nrep} tensors; total {time.time()-t_start:.0f}s", flush=True)

if __name__ == "__main__": main()
