#!/usr/bin/env python3
"""
Train a THERMAL-SPECIFIC ReID embedding on individual deer, and score it honestly.

The requested cues map onto this model as follows:
  body shape, thermal silhouette   the 64x64 crop itself — what the CNN sees
  relative size                    native box w/h, injected as 2 extra input channels so
                                   the embedding can use scale without the resize erasing it
  movement pattern                 NOT here. It is already 3 of the temporal head's 10
                                   per-timestep features (dx, dy, dt) and belongs there,
                                   not in a per-crop appearance encoder.
  antler structure                 not resolvable: ~19 px per metre of animal puts a tine
                                   under half a pixel, and the corpus spans Dec-Jan when
                                   bucks shed. Recorded as a limitation, not modelled.

PROTOCOL — open-set, which is the only honest one here.
Identities come from CVAT tracks, and every deer appears in exactly ONE video, so the
video split doubles as an identity split: the model trains on the ~152 deer in the
detector's training videos and is scored on deer it has never seen, in videos it has
never seen. That is the correct ReID evaluation (unseen identities), and it is also the
only one available.

Scoring is rank-1 with a TEMPORAL gallery/query split inside each held-out video —
identical to src/reid/reid_feasibility.py, so the trained model's number sits directly
beside the ImageNet-ResNet50 and geometry-only baselines from that job.

CEILING CAVEAT, stated up front: within-video means gallery and query share range,
lighting, pose and thermal calibration. No deer in this corpus appears in two videos, so
cross-session identification cannot be measured at all. A good number here does NOT
establish that the model re-identifies individuals across sightings.

Exports ONNX so the encoder drops straight into BoT-SORT (`model: <path>.onnx` in the
tracker yaml, replacing `auto`).

Usage:
  python src/reid/train_thermal_reid.py --crops <dir> --out results/reid/thermal
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "temporal"))
from dataset import video_splits                                       # noqa: E402

SIZE = 64


# --------------------------------------------------------------------------- data
def load_split(crop_dir: str, splits: dict, want: tuple[str, ...]) -> list[dict]:
    """-> [{video, crops, ident, frame, box}] for videos in the requested split(s)."""
    out = []
    for f in sorted(glob.glob(os.path.join(crop_dir, "*.npz"))):
        video = os.path.splitext(os.path.basename(f))[0]
        if splits.get(video) not in want:
            continue
        d = np.load(f)
        out.append({"video": video, "crops": d["crops"], "ident": d["ident"],
                    "frame": d["frame"], "box": d["box"]})
    return out


def to_tensor(crops: np.ndarray, box: np.ndarray) -> torch.Tensor:
    """[N,64,64] uint8 + [N,2] -> [N,3,64,64]: image, plus w and h as constant planes.

    The resize to a fixed 64x64 destroys absolute scale, and scale is one of the cues we
    were asked to use. Feeding native w/h as their own channels puts it back without the
    network having to infer it from a stretched image.
    """
    x = torch.from_numpy(crops.astype(np.float32) / 255.0).unsqueeze(1)
    wh = torch.from_numpy(box.astype(np.float32) / 100.0)              # ~0-1 for real deer
    planes = wh.view(-1, 2, 1, 1).expand(-1, 2, SIZE, SIZE)
    return torch.cat([x, planes], dim=1)


class Encoder(nn.Module):
    """Small ResNet-ish trunk. Deliberately tiny: ~150 training identities cannot support
    a full ResNet-50 without memorising, the same capacity lesson as §6.5."""

    def __init__(self, dim: int = 128):
        super().__init__()
        def blk(i, o, s=2):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))
        self.trunk = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 128), blk(128, 128))
        self.head = nn.Linear(128, dim)

    def forward(self, x):
        f = self.trunk(x).mean(dim=(2, 3))
        return F.normalize(self.head(f), dim=1)


def batch_hard_triplet(emb: torch.Tensor, lab: torch.Tensor, margin: float = 0.3):
    """Standard batch-hard: hardest positive vs hardest negative per anchor."""
    d = torch.cdist(emb, emb)
    same = lab[:, None] == lab[None, :]
    eye = torch.eye(len(lab), dtype=torch.bool, device=lab.device)
    pos = torch.where(same & ~eye, d, torch.full_like(d, -1.0)).max(dim=1).values
    neg = torch.where(~same, d, torch.full_like(d, 1e9)).min(dim=1).values
    ok = pos >= 0                                                      # anchor had a positive
    if ok.sum() == 0:
        return emb.sum() * 0.0
    return F.relu(pos[ok] - neg[ok] + margin).mean()


# --------------------------------------------------------------------------- eval
def rank1_temporal(model, videos: list[dict], device, min_per_half: int = 4,
                   min_ident: int = 3) -> tuple[float, int, list]:
    """Same protocol as reid_feasibility.py so numbers are directly comparable."""
    model.eval()
    rows, num, den = [], 0.0, 0
    with torch.no_grad():
        for v in videos:
            gal, qry = [], []
            for ti in np.unique(v["ident"]):
                m = v["ident"] == ti
                order = np.argsort(v["frame"][m])
                c, b = v["crops"][m][order], v["box"][m][order]
                if len(c) < 2 * min_per_half:
                    continue
                h = len(c) // 2
                ea = model(to_tensor(c[:h], b[:h]).to(device)).mean(0)
                eb = model(to_tensor(c[h:], b[h:]).to(device)).mean(0)
                gal.append(ea); qry.append(eb)
            if len(gal) < min_ident:
                continue
            g = F.normalize(torch.stack(gal), dim=1)
            q = F.normalize(torch.stack(qry), dim=1)
            hit = ((q @ g.T).argmax(1) == torch.arange(len(q), device=device)).float().mean()
            rows.append({"video": v["video"], "identities": len(gal),
                         "rank1": round(float(hit), 4), "chance": round(1 / len(gal), 4)})
            num += float(hit) * len(gal); den += len(gal)
    model.train()
    return (num / den if den else 0.0), den, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", required=True)
    ap.add_argument("--out", default="results/reid/thermal")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--p-ident", type=int, default=16, help="identities per batch")
    ap.add_argument("--k-inst", type=int, default=4, help="crops per identity per batch")
    ap.add_argument("--iters", type=int, default=100, help="batches per epoch")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    splits = video_splits()
    tr = load_split(args.crops, splits, ("train",))
    ev = load_split(args.crops, splits, ("val", "test"))
    if not tr or not ev:
        raise SystemExit(f"train={len(tr)} eval={len(ev)} videos — check --crops")

    # global identity id = (video, cvat track); every deer lives in exactly one video
    pool: dict[int, tuple] = {}
    gid = 0
    for v in tr:
        for ti in np.unique(v["ident"]):
            m = v["ident"] == ti
            if m.sum() < args.k_inst:
                continue
            pool[gid] = (v["crops"][m], v["box"][m]); gid += 1
    if len(pool) < args.p_ident:
        raise SystemExit(f"only {len(pool)} trainable identities")
    print(f"train identities {len(pool)} over {len(tr)} videos | "
          f"eval videos {len(ev)}", flush=True)

    model = Encoder(args.dim).to(device)
    nid = len(pool)
    classifier = nn.Linear(args.dim, nid).to(device)                   # ID loss stabilises
    opt = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()),
                            lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best, hist = -1.0, []
    for ep in range(1, args.epochs + 1):
        tot = 0.0
        for _ in range(args.iters):
            ids = np.random.choice(nid, args.p_ident, replace=False)
            xs, ys = [], []
            for g in ids:
                c, b = pool[g]
                sel = np.random.choice(len(c), args.k_inst,
                                       replace=len(c) < args.k_inst)
                xs.append(to_tensor(c[sel], b[sel])); ys += [g] * args.k_inst
            x = torch.cat(xs).to(device)
            y = torch.as_tensor(ys, device=device)
            # light augmentation: horizontal flip only. A deer's left and right sides are
            # both valid views; rotation/crop would fabricate poses the sensor never sees.
            if np.random.rand() < 0.5:
                x = torch.flip(x, dims=[3])
            e = model(x)
            loss = batch_hard_triplet(e, y) + 0.5 * F.cross_entropy(classifier(e), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        sched.step()

        if ep % 5 == 0 or ep == args.epochs:
            r1, nde, rows = rank1_temporal(model, ev, device)
            hist.append({"epoch": ep, "loss": round(tot / args.iters, 4),
                         "rank1_unseen": round(r1, 4), "identities": nde})
            print(f"ep {ep:>3}  loss {tot/args.iters:.4f}  "
                  f"rank-1 (unseen ident) {r1:.3f} over {nde} deer", flush=True)
            if r1 > best:
                best = r1
                torch.save({"model": model.state_dict(), "dim": args.dim},
                           os.path.join(args.out, "best.pt"))
                json.dump(rows, open(os.path.join(args.out, "per_video.json"), "w"),
                          indent=1)

    # ---- export for BoT-SORT (`model: <this>.onnx` replaces `auto`) ----
    model.load_state_dict(torch.load(os.path.join(args.out, "best.pt"))["model"])
    model.eval()
    onnx_path = os.path.join(args.out, "thermal_reid.onnx")
    torch.onnx.export(model, torch.zeros(1, 3, SIZE, SIZE, device=device), onnx_path,
                      input_names=["images"], output_names=["features"],
                      dynamic_axes={"images": {0: "batch"}, "features": {0: "batch"}},
                      opset_version=17)
    json.dump(hist, open(os.path.join(args.out, "history.json"), "w"), indent=1)

    print(f"\nBEST rank-1 on UNSEEN identities: {best:.3f}")
    print(f"-> {args.out}/  (best.pt, thermal_reid.onnx, per_video.json, history.json)")
    print("Compare against src/reid/reid_feasibility.py: this must beat the geometry-only "
          "baseline to justify an appearance stage at all. Within-video ceiling — no deer "
          "in this corpus appears in two videos, so cross-session ReID is unmeasurable.")


if __name__ == "__main__":
    main()
