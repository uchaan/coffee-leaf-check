#!/usr/bin/env python
"""Template for gptq_iq.py with some tensors' types changed (2026-10-02): gptq_iq.py takes only types, shapes and byte sizes from the
template for the tensors it quantises and copies every other tensor's bytes, so a retyped tensor gets zero bytes of its new type's size
(the build overwrites them). usage: retype_template.py SRC OUT TYPES_FILE   (TYPES_FILE lines: name=TYPE, only the changed tensors)"""
import sys, numpy as np, gguf
from gguf import GGUFReader, GGUFWriter
src, out, tf = sys.argv[1:4]
new = dict(l.strip().split("=") for l in open(tf) if l.strip())
T = GGUFReader(src); arch = bytes(T.fields["general.architecture"].parts[-1]).decode(); w = GGUFWriter(out, arch=arch, endianess=T.endianess)
for f in T.fields.values():
    if f.name == "general.architecture" or f.name.startswith("GGUF."): continue
    vt = f.types[0]; st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None; w.add_key_value(f.name, f.contents(), vt, sub_type=st)
data = {}
for t in T.tensors:
    if t.name in new:
        qt = gguf.GGMLQuantizationType[new[t.name]]; logical = tuple(int(x) for x in reversed(t.shape))
        bshape = gguf.quant_shape_to_byte_shape(logical, qt); arr = np.zeros(bshape, dtype=np.uint8)
        w.add_tensor_info(t.name, arr.shape, arr.dtype, arr.nbytes, qt); data[t.name] = arr
    else:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for t in T.tensors: w.write_tensor_data(data[t.name] if t.name in data else np.asarray(t.data), tensor_endianess=T.endianess)
w.close(); print(f"wrote {out}: {len(data)} tensors retyped")
