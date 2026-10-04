"""CNN baseline in the BRACOL paper's setup (Esgario et al. 2020: ImageNet ResNet50, transfer learning, flips/rotation/colour
augmentation) trained on OUR splits, for a same-data comparison with the fine-tuned VLM.
--split hack   : train = BRACOL dev (val = the same 82 dev images the adapters use), test = BRACOL test 1,266
--split hackjm : hack + the 559 JMuBEN training originals (../data/jmuben_train_originals.csv) in train
--split s2     : train = dev + 60 % of test (1,179, paper-sized; val = int(sha,16) % 10 == 0 of it), test = the other 506
Images: the protocol 512 px JPEGs, resized to 224 for the network. usage: cnn_baseline.py --split hack|hackjm|s2 --out DIR"""
import argparse, csv, json, os, random
import numpy as np, torch, torchvision
from PIL import Image
from torchvision import transforms as T

H = os.environ.get("WORK", "work"); CACHE = os.environ.get("HN04B_CACHE", f"{H}/bracol/cache512")                                   # data, models and outputs (see ../agentic-quantization/loop/env.example.sh)
AQ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../agentic-quantization")   # manifest, prompt, harness
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data"); L = "ABCDE"
ap = argparse.ArgumentParser(); ap.add_argument("--split", required=True, choices=["hack", "hackjm", "s2"]); ap.add_argument("--out", required=True); ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--size", type=int, default=224); a = ap.parse_args(); torch.manual_seed(0); random.seed(0); np.random.seed(0)
rows = list(csv.DictReader(open(f"{AQ}/manifest.csv")))
if a.split == "hack":
    dev = [r for r in rows if r["split"] == "dev"]; va = [r for r in dev if int(r["sha256"], 16) % 5 == 0]; tr = [r for r in dev if int(r["sha256"], 16) % 5 != 0]
    te = [r for r in rows if r["split"] == "test"]
elif a.split == "hackjm":
    dev = [r for r in rows if r["split"] == "dev"]; va = [r for r in dev if int(r["sha256"], 16) % 5 == 0]; tr = [r for r in dev if int(r["sha256"], 16) % 5 != 0]
    tr = tr + [dict(r, jm=1) for r in csv.DictReader(open(f"{DATA}/jmuben_train_originals.csv"))]; te = [r for r in rows if r["split"] == "test"]
else:
    S = list(csv.DictReader(open(f"{DATA}/bracol_paper_sized_split.csv"))); pool = [r for r in S if r["split"] == "dev"]; te = [r for r in S if r["split"] == "test"]
    va = [r for r in pool if int(r["sha256"], 16) % 10 == 0]; tr = [r for r in pool if int(r["sha256"], 16) % 10 != 0]
print(f"split {a.split}: train {len(tr)} val {len(va)} test {len(te)}", flush=True)
norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
ttr = T.Compose([T.Resize((a.size, a.size)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(90), T.ColorJitter(0.2, 0.2, 0.2, 0.05), T.ToTensor(), norm])
tev = T.Compose([T.Resize((a.size, a.size)), T.ToTensor(), norm])
img = {r["sha256"]: Image.open(f"{H}/jmuben/{r['path']}" if r.get("jm") else f"{CACHE}/{r['sha256']}.jpg").convert("RGB") for r in tr + va + te}
class DS(torch.utils.data.Dataset):
    def __init__(s, R, tf): s.R, s.tf = R, tf
    def __len__(s): return len(s.R)
    def __getitem__(s, i): r = s.R[i]; return s.tf(img[r["sha256"]]), L.index(r["label_letter"])
dl = lambda R, tf, sh: torch.utils.data.DataLoader(DS(R, tf), batch_size=32, shuffle=sh, num_workers=4)
m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2); m.fc = torch.nn.Linear(m.fc.in_features, 5); m = m.cuda()
opt = torch.optim.Adam(m.parameters(), lr=1e-4, weight_decay=1e-4); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
def evaluate(R):
    m.eval(); P = []
    with torch.no_grad():
        for x, _ in dl(R, tev, False): P.append(m(x.cuda()).argmax(1).cpu())
    p = [L[i] for i in torch.cat(P).tolist()]; y = [r["label_letter"] for r in R]; pr, rc, f1 = [], [], []
    for c in L:
        tp = sum(u == c and v == c for u, v in zip(p, y)); fp = sum(u == c and v != c for u, v in zip(p, y)); fn = sum(u != c and v == c for u, v in zip(p, y))
        a1 = tp / (tp + fp) if tp + fp else 0; b1 = tp / (tp + fn) if tp + fn else 0; pr.append(a1); rc.append(b1); f1.append(2 * a1 * b1 / (a1 + b1) if a1 + b1 else 0)
    return dict(acc=float(np.mean([u == v for u, v in zip(p, y)])), macro_p=float(np.mean(pr)), macro_r=float(np.mean(rc)), macro_f1=float(np.mean(f1)), per_class_f1=dict(zip(L, [round(x, 3) for x in f1])))
best = (-1, None); os.makedirs(a.out, exist_ok=True)
for ep in range(a.epochs):
    m.train()
    for x, y in dl(tr, ttr, True):
        opt.zero_grad(); loss = torch.nn.functional.cross_entropy(m(x.cuda()), y.cuda()); loss.backward(); opt.step()
    sch.step(); v = evaluate(va)
    if v["macro_f1"] > best[0]: best = (v["macro_f1"], ep + 1); torch.save(m.state_dict(), f"{a.out}/best.pt")
    print(f"epoch {ep+1}: val acc {v['acc']:.4f} F1 {v['macro_f1']:.4f}", flush=True)
m.load_state_dict(torch.load(f"{a.out}/best.pt")); t = evaluate(te)
res = dict(split=a.split, train=len(tr), val=len(va), n_test=len(te), best_epoch=best[1], val_f1=best[0], test=t, params=sum(p.numel() for p in m.parameters()))
json.dump(res, open(f"{a.out}/result.json", "w"), indent=1); print("TEST", json.dumps(res), flush=True)
