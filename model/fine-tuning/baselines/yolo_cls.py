"""YOLO11 classification baseline on our BRACOL splits; trains on $WORK/yolo/<SPLIT> (make_yolo_dirs.py), tests on BRACOL test and
writes $WORK/yolo/runs/<SPLIT>-<MODEL>-<IMGSZ>/result.json. usage: yolo_cls.py SPLIT(hack|hackjm) MODEL(yolo11n-cls|yolo11s-cls|yolo11m-cls) IMGSZ"""
import json, os, sys
import numpy as np
os.environ["YOLO_OFFLINE"] = "0"                                    # ultralytics downloads the ImageNet yolo11*-cls.pt weights on first use
from ultralytics import YOLO
H = os.environ.get("WORK", "work")                                   # data, models and outputs (see env.example.sh at the repository root)
split, mname, imgsz = sys.argv[1], sys.argv[2], int(sys.argv[3]); L = "ABCDE"
name = f"{split}-{mname}-{imgsz}"; m = YOLO(f"{mname}.pt")
m.train(data=f"{H}/yolo/{split}", epochs=60, imgsz=imgsz, batch=32, project=f"{H}/yolo/runs", name=name, exist_ok=True, seed=0, deterministic=True,
        fliplr=0.5, flipud=0.5, degrees=90, hsv_h=0.015, hsv_s=0.4, hsv_v=0.3, patience=100, verbose=False, plots=False, workers=4)
best = YOLO(f"{H}/yolo/runs/{name}/weights/best.pt"); y, p = [], []
for c in L:
    d = f"{H}/yolo/{split}/test/{c}"
    for f in sorted(os.listdir(d)):
        r = best.predict(f"{d}/{f}", imgsz=imgsz, verbose=False)[0]; p.append(r.names[int(r.probs.top1)]); y.append(c)
pr, rc, f1 = [], [], []
for c in L:
    tp = sum(u == c and v == c for u, v in zip(p, y)); fp = sum(u == c and v != c for u, v in zip(p, y)); fn = sum(u != c and v == c for u, v in zip(p, y))
    a1 = tp / (tp + fp) if tp + fp else 0; b1 = tp / (tp + fn) if tp + fn else 0; pr.append(a1); rc.append(b1); f1.append(2 * a1 * b1 / (a1 + b1) if a1 + b1 else 0)
res = dict(split=split, model=mname, imgsz=imgsz, n_test=len(y), acc=float(np.mean([u == v for u, v in zip(p, y)])), macro_p=float(np.mean(pr)), macro_r=float(np.mean(rc)),
           macro_f1=float(np.mean(f1)), per_class_f1=dict(zip(L, [round(x, 3) for x in f1])), weights_bytes=os.path.getsize(f"{H}/yolo/runs/{name}/weights/best.pt"),
           params=sum(q.numel() for q in best.model.parameters()))
json.dump(res, open(f"{H}/yolo/runs/{name}/result.json", "w"), indent=1); print("TEST", json.dumps(res), flush=True)
