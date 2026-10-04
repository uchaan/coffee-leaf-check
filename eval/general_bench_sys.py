#!/usr/bin/env python3
"""Control for eval/general_bench.py (4090): identical items, request and scoring, plus one system line,
"Answer with one letter only." (the instruction the BRACOL prompt carries). Tests whether ours-vs-Unsloth gaps on
the general benchmarks survive when every file is told the answer format up front.
  python eval/general_bench_sys.py run --model ... --lm ... --mmproj ... --cache ... --out ... [--ref ...] --gpu N --port P
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import general_bench as G  # noqa: E402

SYSTEM = "Answer with one letter only."
_body = G.body


def body_with_system(it):
    b = _body(it)
    b["messages"] = [{"role": "system", "content": SYSTEM}] + b["messages"]
    return b


G.body = body_with_system

if __name__ == "__main__":
    G.main()
