"""One-sentence explanation demo: for the first BRACOL test image of each class (data/bracol_paper_sized_split.csv), ask for the
letter (frozen prompt, grammar [A-F]), then ask for one sentence on what is visible, once with the LoRA adapter off (scale 0) and once on.
Expects a running llama-server with the adapter loaded via --lora, e.g.
  $LLAMA_CPP_DIR/build/bin/llama-server -m BASE.gguf --mmproj MMPROJ.gguf --lora LORA.gguf --jinja -c 8192 --port 9899
and the 512 px BRACOL images in $HN04B_CACHE/<sha256>.jpg (data/prep_images.py).
usage: explain_demo.py [--port 9899]"""
import argparse, base64, csv, json, os, requests, sys
ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 9899))); a = ap.parse_args()
U = f"http://127.0.0.1:{a.port}/v1/chat/completions"; H = os.environ.get("WORK", "work")      # data, models and outputs (see ../agentic-quantization/loop/env.example.sh)
CACHE = os.environ.get("HN04B_CACHE", f"{H}/bracol/cache512")                                  # 512 px BRACOL images by sha256
AQ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../agentic-quantization")   # manifest, prompt, harness
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
txt = open(f"{AQ}/protocol/prompt_v1.txt").read(); sys_t = txt.split("### system\n")[1].split("### user\n")[0].strip(); usr_t = txt.split("### user\n")[1].split("### grammar")[0].strip()
NAMES = {"A": "healthy", "B": "coffee leaf rust", "C": "brown eye spot (Cercospora)", "D": "Phoma leaf spot", "E": "leaf miner", "F": "not sure"}
rows = [r for r in csv.DictReader(open(f"{DATA}/bracol_paper_sized_split.csv")) if r["split"] == "test"]
pick = []
for c in "ABCDE": pick += [r for r in rows if r["label_letter"] == c][:1]
for r in pick:
    b64 = base64.b64encode(open(f"{CACHE}/{r['sha256']}.jpg", "rb").read()).decode(); img = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}
    body = {"messages": [{"role": "system", "content": sys_t}, {"role": "user", "content": [img, {"type": "text", "text": usr_t}]}], "max_tokens": 1, "temperature": 0, "grammar": "root ::= [A-F]",
            "chat_template_kwargs": {"enable_thinking": False}}
    ans = requests.post(U, json=body).json()["choices"][0]["message"]["content"].split("</think>")[-1].strip()
    q = (f"Diagnosis: {NAMES.get(ans, ans)}. In one sentence, describe only what you can see on this leaf (colour, shape and position of any spots or damage) that is consistent with this diagnosis.")
    for sc in (0.0, 1.0):
        b2 = {"messages": [{"role": "user", "content": [img, {"type": "text", "text": q}]}], "max_tokens": 90, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}, "lora": [{"id": 0, "scale": sc}]}
        out = requests.post(U, json=b2).json()["choices"][0]["message"]["content"].split("</think>")[-1].strip().replace("\n", " ")
        print(f"[true {r['label_letter']} | pred {ans}] adapter={sc}: {out}")
