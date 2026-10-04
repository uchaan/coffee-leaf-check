"""CNN / YOLO on the grouped JMuBEN test split (the 240 held-out originals, split=test in ../data/jmuben_grouped_split.csv) and on
BRACOL test (1,266). Expects the five trained runs: $WORK/ft/cnn-hack, $WORK/ft/cnn-hackjm (cnn_baseline.py) and
$WORK/yolo/runs/{hack-yolo11m-cls-384, hackjm-yolo11m-cls-384, hackjm-yolo11n-cls-224} (yolo_cls.py). usage: eval_cnn.py"""
import csv, json, os, sys, numpy as np, torch, torchvision
from PIL import Image
from torchvision import transforms as T
H = os.environ.get("WORK", "work"); CACHE = os.environ.get("HN04B_CACHE", f"{H}/bracol/cache512")                                   # data, models and outputs (see env.example.sh at the repository root)
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")   # manifest, prompt, harness
DATA = os.path.join(REPO, "data"); L = "ABCDE"
JM = [(f"{H}/jmuben/{r['path']}", r["label_letter"]) for r in csv.DictReader(open(f"{DATA}/jmuben_grouped_split.csv")) if r["split"] == "test"]
BR = [(f"{CACHE}/{r['sha256']}.jpg", r["label_letter"]) for r in csv.DictReader(open(f"{REPO}/data/manifest.csv")) if r["split"] == "test"]
def score(p, y):
    f1 = []
    for c in L:
        tp = sum(u == c and v == c for u, v in zip(p, y)); fp = sum(u == c and v != c for u, v in zip(p, y)); fn = sum(u != c and v == c for u, v in zip(p, y))
        a = tp / (tp + fp) if tp + fp else 0; b = tp / (tp + fn) if tp + fn else 0; f1.append(2 * a * b / (a + b) if a + b else 0)
    return float(np.mean([u == v for u, v in zip(p, y)])), float(np.mean(f1))
tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
def resnet(ck, data):
    m = torchvision.models.resnet50(); m.fc = torch.nn.Linear(m.fc.in_features, 5); m.load_state_dict(torch.load(ck)); m = m.cuda().eval(); P = []
    with torch.no_grad():
        for i in range(0, len(data), 64): P += m(torch.stack([tf(Image.open(p).convert("RGB")) for p, _ in data[i:i+64]]).cuda()).argmax(1).tolist()
    return [L[i] for i in P]
from ultralytics import YOLO
def yolo(name, sz, data):
    yo = YOLO(f"{H}/yolo/runs/{name}/weights/best.pt"); p = []
    for i in range(0, len(data), 64):
        for res in yo.predict([q for q, _ in data[i:i+64]], imgsz=sz, verbose=False): p.append(res.names[int(res.probs.top1)])
    return p
for name, fn in [("ResNet50 BRACOL-only", lambda d: resnet(f"{H}/ft/cnn-hack/best.pt", d)), ("ResNet50 BRACOL+JMuBEN", lambda d: resnet(f"{H}/ft/cnn-hackjm/best.pt", d)),
                 ("YOLO11m BRACOL-only", lambda d: yolo("hack-yolo11m-cls-384", 384, d)), ("YOLO11m BRACOL+JMuBEN", lambda d: yolo("hackjm-yolo11m-cls-384", 384, d)),
                 ("YOLO11n BRACOL+JMuBEN", lambda d: yolo("hackjm-yolo11n-cls-224", 224, d))]:
    aj, fj = score(fn(JM), [y for _, y in JM]); ab, fb = score(fn(BR), [y for _, y in BR])
    print(f"{name:26s} JMuBEN-240 acc {aj:.3f} F1 {fj:.3f} | BRACOL test acc {ab:.3f} F1 {fb:.3f}", flush=True)
