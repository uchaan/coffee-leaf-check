#!/usr/bin/env python3
"""CPU-proxy queue: for every job of one model, harness `cpu` pinned to fixed cores with taskset.
Re-reads the jobs CSV every pass. Stops when <runs>/STOP_cpu_<model> exists and nothing is left.
  python eval/cpu_queue.py --model Qwen3.5-2B --cores 32-35 --port 8190 --prompt protocol/prompt_v1.txt \
      --jobs jobs.csv --mmproj mmproj.gguf
"""
import argparse, csv, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from queue_common import RUNS, tag_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--cores", required=True)
    ap.add_argument("--port", required=True); ap.add_argument("--prompt", required=True)
    ap.add_argument("--jobs", required=True); ap.add_argument("--mmproj", required=True)
    ap.add_argument("--runs", default=RUNS)
    a = ap.parse_args()
    pname = os.path.basename(a.prompt).replace(".txt", "")
    while True:
        jobs = [j for j in csv.DictReader(open(a.jobs)) if j["model"] == a.model]
        pending, ran = 0, False
        for j in jobs:
            if j["source"] == "bf16":  # not a phone candidate; skipped to save CPU time
                continue
            if not os.path.exists(j["lm_path"]):
                pending += 1
                continue
            tag = tag_of(j, pname) + "__cpu4"
            out = os.path.join(a.runs, "cpu", tag)
            if os.path.exists(os.path.join(out, "cpu.json")) or os.path.exists(os.path.join(out, "FAILED")):
                continue
            os.makedirs(out, exist_ok=True)
            cmd = ["taskset", "-c", a.cores, sys.executable, os.path.join(HERE, "harness.py"), "cpu",
                   "--model", a.model, "--lm", j["lm_path"], "--mmproj", a.mmproj, "--prompt", a.prompt,
                   "--port", a.port, "--threads", "4", "--out-root", os.path.join(a.runs, "cpu"), "--tag", tag]
            print(time.strftime("%T"), "CPU", tag, flush=True)
            with open(os.path.join(out, "harness.out"), "w") as lf:
                r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
            if r.returncode != 0:
                open(os.path.join(out, "FAILED"), "w").write(f"returncode {r.returncode}\n")
            ran = True
            break  # re-read the jobs list after every job
        if not ran and pending == 0 and os.path.exists(os.path.join(a.runs, f"STOP_cpu_{a.model}")):
            break
        if not ran:
            time.sleep(30)


if __name__ == "__main__":
    main()
