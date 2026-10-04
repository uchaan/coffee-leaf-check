"""How close are the 240 held-out JMuBEN originals to the training photos? Supports reading the JMuBEN score as a
same-dataset score. Every image is embedded with an ImageNet ResNet50 (no fine-tuning); for each held-out original we
take the cosine similarity to its nearest training image (all distinct images, rotated/flipped copies included, of the
559 training originals) and check whether that neighbour has the same label. BRACOL test vs dev is the reference.
usage: WORK=... python check_jmuben_overlap.py   (expects $WORK/jmuben/images/ and $WORK/bracol/cache512/)"""
import csv, os, sys, numpy as np, torch, torchvision
from PIL import Image
from torchvision import transforms as T
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "../data"))
from make_jmuben_split import group_images, split_groups   # noqa: E402
H = os.environ.get("WORK", "work"); AQ = os.path.join(HERE, "../../agentic-quantization")
R, gid = group_images(f"{H}/jmuben", os.path.join(HERE, "../data/jmuben_distinct_manifest.csv")); split, one = split_groups(R, gid)
test_sha = {one[g]["sha256"] for g in one if split[g] == "test"}
tr = [r for r, g in zip(R, gid) if split[g] == "train"]; te = [r for r in R if r["sha256"] in test_sha]
m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2); m.fc = torch.nn.Identity()
dev = "cuda" if torch.cuda.is_available() else "cpu"; m = m.to(dev).eval()
tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
def emb(paths):
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), 64):
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i+64]]).to(dev); out.append(torch.nn.functional.normalize(m(x), dim=1).cpu())
    return torch.cat(out)
Ete = emb([f"{H}/jmuben/{r['path']}" for r in te]); Etr = emb([f"{H}/jmuben/{r['path']}" for r in tr]); best, idx = (Ete @ Etr.T).max(1)
q = [.05, .25, .5, .75, .95]
print(f"JMuBEN: {len(te)} held-out originals vs {len(tr)} training images; nearest cosine at 5/25/50/75/95 %:", np.round(np.quantile(best.numpy(), q), 3))
print("  nearest training image has the same label:", round(float(np.mean([tr[int(j)]["label_letter"] == te[i]["label_letter"] for i, j in enumerate(idx)])), 3))
B = list(csv.DictReader(open(f"{AQ}/manifest.csv")))
bd = emb([f"{H}/bracol/cache512/{r['sha256']}.jpg" for r in B if r["split"] == "dev"]); bt = emb([f"{H}/bracol/cache512/{r['sha256']}.jpg" for r in B if r["split"] == "test"])
print("BRACOL test vs dev (reference), nearest cosine at 5/25/50/75/95 %:", np.round(np.quantile((bt @ bd.T).max(1).values.numpy(), q), 3))
