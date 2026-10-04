#!/usr/bin/env python3
"""GPU worker. Models are served in priority order (e.g. Qwen3.5-2B first, then Qwen3.5-0.8B). After every
job it re-reads the jobs CSV and takes the first job (BF16 reference first, then file order) that is not
done and not claimed by another worker, so several workers (one per GPU) can share one jobs list and
jobs can be added or reordered while they run. Results do not depend on the GPU: the same file gave
bit-identical probabilities on two different GPUs.

jobs CSV columns: model,source,quant_name,lm_path,calibration
A job = harness run --split all (scored against the BF16 rows of the same prompt), then text KLD.
mmproj and BF16 per model: <a100-dir>/<model>/mmproj-<model>-Q8_0.gguf and <model>-BF16.gguf.
Stops when <runs>/STOP exists and nothing runnable is left.
  python benchmark/gpu_queue.py --models Qwen3.5-2B,Qwen3.5-0.8B --gpu 0 --port 8090 \
      --prompt protocol/prompt_v1.txt --jobs $WORK/jobs.csv --a100-dir $WORK/models/a100
"""
import argparse, csv, os, shutil, socket, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from queue_common import RUNS, WORK, tag_of  # noqa: E402


def claim(path):
    try:
        os.mkdir(path)
        open(os.path.join(path, "by"), "w").write(f"{socket.gethostname()} pid {os.getpid()} {time.strftime('%F %T')}\n")
        return True
    except FileExistsError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma-separated, in priority order")
    ap.add_argument("--gpu", required=True); ap.add_argument("--port", required=True)
    ap.add_argument("--prompt", required=True); ap.add_argument("--jobs", required=True)
    ap.add_argument("--a100-dir", required=True)
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--no-textkld", action="store_true")
    ap.add_argument("--dataset", choices=["bracol", "jmuben"], default="bracol",
                    help="jmuben: optional extra test set (test split only, tag suffix __jmuben, no text KLD)")
    a = ap.parse_args()
    if a.dataset == "jmuben":
        a.no_textkld = True
    models = a.models.split(",")
    pname = os.path.basename(a.prompt).replace(".txt", "") + ("__jmuben" if a.dataset == "jmuben" else "")
    kld_dir = os.path.join(a.runs, "textkld"); os.makedirs(kld_dir, exist_ok=True)
    claims = os.path.join(a.runs, "claims"); os.makedirs(claims, exist_ok=True)
    while True:
        alljobs = list(csv.DictReader(open(a.jobs)))
        todo = None
        for model in models:
            jobs = [j for j in alljobs if j["model"] == model]
            bf = [j for j in jobs if j["source"] == "bf16"]
            ref = os.path.join(a.runs, tag_of(bf[0], pname), "images.csv") if bf else None
            for j in sorted(jobs, key=lambda j: j["source"] != "bf16"):
                if not os.path.exists(j["lm_path"]):
                    continue
                tag = tag_of(j, pname)
                out = os.path.join(a.runs, tag)
                stem = os.path.basename(j["lm_path"]).replace(".gguf", "")
                kout = os.path.join(kld_dir, f"{model}__{stem}.json")
                need_run = not (os.path.exists(os.path.join(out, "metrics.json")) or os.path.exists(os.path.join(out, "FAILED")))
                need_kld = not a.no_textkld and j["source"] != "bf16" and not os.path.exists(kout)
                if need_run and j["source"] != "bf16" and (ref is None or not os.path.exists(ref)):
                    need_run = False  # wait for the BF16 reference
                if need_run and not claim(os.path.join(claims, tag)):
                    need_run = False
                if need_kld and not claim(os.path.join(claims, "kld__" + os.path.basename(kout))):
                    need_kld = False
                if need_run or need_kld:
                    todo = (model, j, out, kout, need_run, need_kld, ref)
                    break
            if todo:
                break
        if todo is None:
            if os.path.exists(os.path.join(a.runs, "STOP")):
                break
            time.sleep(20)
            continue
        model, j, out, kout, need_run, need_kld, ref = todo
        tag = os.path.basename(out)
        mmproj = os.path.join(a.a100_dir, model, f"mmproj-{model}-Q8_0.gguf")
        bf16 = os.path.join(a.a100_dir, model, f"{model}-BF16.gguf")
        if need_run:
            st0 = os.stat(j["lm_path"])
            cmd = [sys.executable, os.path.join(HERE, "harness.py"), "run", "--model", model,
                   "--lm", j["lm_path"], "--mmproj", mmproj, "--prompt", a.prompt, "--split", "all",
                   "--gpu", a.gpu, "--port", a.port, "--tag", tag]
            if a.dataset == "jmuben":
                cmd += ["--manifest", os.path.join(os.path.dirname(HERE), "manifest_jmuben.csv"),
                        "--data-root", os.environ.get("JMUBEN_ROOT", os.path.join(WORK, "jmuben", "raw"))]
            if j["source"] != "bf16":
                cmd += ["--ref", ref]
            print(time.strftime("%T"), "RUN", tag, flush=True)
            os.makedirs(out, exist_ok=True)
            try:
                with open(os.path.join(out, "harness.out"), "w") as lf:
                    r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
                rc = r.returncode
            except OSError:
                rc = -1
            if rc != 0:
                try:
                    st1 = os.stat(j["lm_path"])
                    changed = (st1.st_mtime_ns, st1.st_size) != (st0.st_mtime_ns, st0.st_size)
                except FileNotFoundError:
                    changed = True
                if changed:  # the file was replaced while it ran: drop this attempt, it runs again
                    shutil.rmtree(out, ignore_errors=True)
                    shutil.rmtree(os.path.join(claims, tag), ignore_errors=True)
                    print(time.strftime("%T"), "RETRY (file replaced)", tag, flush=True)
                else:
                    os.makedirs(out, exist_ok=True)
                    open(os.path.join(out, "FAILED"), "w").write(f"returncode {rc}\n")
                    print(time.strftime("%T"), "FAILED", tag, flush=True)
        if need_kld:
            print(time.strftime("%T"), "TEXTKLD", os.path.basename(j["lm_path"]), flush=True)
            subprocess.run([sys.executable, os.path.join(HERE, "harness.py"), "textkld", "--bf16", bf16,
                            "--lm", j["lm_path"], "--gpu", a.gpu, "--out", kout, "--log", kout + ".log"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
