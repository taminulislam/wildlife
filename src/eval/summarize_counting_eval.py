#!/usr/bin/env python3
"""
Aggregate every counting_detection_eval.py run into ONE paper-ready table.

Reads results/counting_eval/*.csv (written per model x GT-set x confidence) and emits:
  * counting_eval_ALL.csv   — tidy long form, every row
  * counting_eval_TABLE.md  — markdown tables grouped for the paper

Usage: python src/eval/summarize_counting_eval.py [--dir results/counting_eval]
"""
from __future__ import annotations
import argparse
import csv
import glob
import os
from collections import defaultdict

ORDER = ["yolov8m", "yolov9m", "yolov10m", "yolo11m", "yolo12m", "rtdetr-l"]
CRIT_LABEL = {
    "iou50": "IoU>=0.50 (standard)",
    "iou30": "IoU>=0.30",
    "touch": "any overlap (counting)",
    "center": "centre-in-box",
}


def model_sort(name: str) -> int:
    for i, m in enumerate(ORDER):
        if name.startswith(m):
            return i
    return len(ORDER)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/counting_eval")
    ap.add_argument("--out-prefix", default="results/counting_eval/counting_eval")
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.join(args.dir, "*.csv"))):
        if os.path.basename(f).startswith("counting_eval_"):
            continue                      # don't re-read our own outputs
        with open(f) as fh:
            for r in csv.DictReader(fh):
                tag = r["tag"]
                if tag.endswith("_KEYFRAME"):
                    gt, model = "keyframe", tag[: -len("_KEYFRAME")]
                elif tag.endswith("_FULLGT"):
                    gt, model = "full", tag[: -len("_FULLGT")]
                else:
                    gt, model = "full", tag
                r["gt_set"] = gt
                r["model"] = model.replace("_640", "").replace("_1280", "")
                r["imgsz"] = "1280" if "_1280" in model else "640"
                rows.append(r)

    if not rows:
        raise SystemExit(f"no result CSVs under {args.dir}")

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    cols = ["model", "imgsz", "gt_set", "conf", "criterion",
            "precision", "recall", "f1", "tp", "fp", "fn"]
    with open(f"{args.out_prefix}_ALL.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["gt_set"], r["conf"],
                                             model_sort(r["model"]), r["criterion"])):
            w.writerow(r)

    # ---- markdown: one table per (gt_set, conf), models x criteria ----
    grp = defaultdict(dict)
    for r in rows:
        grp[(r["gt_set"], r["conf"])][(r["model"], r["criterion"])] = r

    lines = ["# Detection results — counting-oriented evaluation", "",
             "P / R / F1 for every model under four matching rules.", "",
             "* **full** GT = all test boxes (94% CVAT-interpolated).",
             "* **keyframe** GT = only boxes a human placed (182 imgs / 361 boxes).",
             "* `IoU>=0.50` is the standard detection metric — publish this in the "
             "detection table.",
             "* `any overlap` is the counting criterion — report as "
             "**presence/counting recall**, never as mAP.", ""]

    for (gt, conf) in sorted(grp, key=lambda k: (k[0] != "keyframe", float(k[1]))):
        cell = grp[(gt, conf)]
        models = sorted({m for m, _ in cell}, key=model_sort)
        lines += [f"## GT = {gt}, confidence = {conf}", "",
                  "| Model | " + " | ".join(
                      f"{CRIT_LABEL[c]}<br>P / R / F1" for c in CRIT_LABEL) + " |",
                  "|---" * (len(CRIT_LABEL) + 1) + "|"]
        for m in models:
            cells = []
            for c in CRIT_LABEL:
                r = cell.get((m, c))
                cells.append(f"{float(r['precision']):.3f} / {float(r['recall']):.3f} / "
                             f"{float(r['f1']):.3f}" if r else "—")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")

    with open(f"{args.out_prefix}_TABLE.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out_prefix}_ALL.csv  ({len(rows)} rows)")
    print(f"wrote {args.out_prefix}_TABLE.md")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
