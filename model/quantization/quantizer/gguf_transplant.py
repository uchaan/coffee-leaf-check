#!/usr/bin/env python
"""Byte-level tensor transplant between two GGUF files of the same model (no re-quantisation).

The output has BASE's metadata and tensor order; every tensor named in --take is copied (type and bytes) from DONOR
instead. Used to build mixed-allocation hybrids from two quantised builds, e.g. a build on a smaller allocation
(base) upgraded group by group with the tensors of a build on a larger allocation (donor).

usage: gguf_transplant.py BASE DONOR OUT (--take NAME [NAME ...] | --take-file FILE) [--dry-run]
"""
import argparse, sys
import numpy as np, gguf
from gguf import GGUFReader, GGUFWriter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("donor"); ap.add_argument("out")
    ap.add_argument("--take", nargs="*", default=[]); ap.add_argument("--take-file")
    ap.add_argument("--dry-run", action="store_true", help="only print the resulting byte count")
    a = ap.parse_args()
    take = set(a.take)
    if a.take_file:
        take |= {l.strip() for l in open(a.take_file) if l.strip()}
    R = GGUFReader(a.base); D = {t.name: t for t in GGUFReader(a.donor).tensors}
    missing = sorted(n for n in take if n not in D or n not in {t.name for t in R.tensors})
    if missing:
        sys.exit(f"tensors not in both files: {missing[:5]} ... ({len(missing)})")
    srcs = [D[t.name] if t.name in take else t for t in R.tensors]
    for t, s in zip(R.tensors, srcs):
        if s is not t and list(s.shape) != list(t.shape):
            sys.exit(f"shape mismatch for {t.name}: {list(t.shape)} vs {list(s.shape)}")
    nbytes = sum(int(s.data.nbytes) for s in srcs)
    delta = nbytes - sum(int(t.data.nbytes) for t in R.tensors)
    print(f"tensors {len(srcs)}, taken {len(take)}, tensor bytes {nbytes:,} (delta {delta:+,})", flush=True)
    if a.dry_run:
        return
    arch = bytes(R.fields["general.architecture"].parts[-1]).decode()
    w = GGUFWriter(a.out, arch=arch, endianess=R.endianess)
    for f in R.fields.values():
        if f.name == "general.architecture" or f.name.startswith("GGUF."):
            continue
        vt = f.types[0]; st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(f.name, f.contents(), vt, sub_type=st)
    for s in srcs:
        w.add_tensor_info(s.name, s.data.shape, s.data.dtype, s.data.nbytes, s.tensor_type)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for s in srcs:
        w.write_tensor_data(np.asarray(s.data).reshape(s.data.shape), tensor_endianess=R.endianess)
    w.close()
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
