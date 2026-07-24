"""
Assemble a YOLO-format dataset from annotated frames, split by SITE (not by frame).

After annotation (e.g. CVAT export to "YOLO 1.1" / Ultralytics format), you have images
and matching ``.txt`` label files. This tool lays them out as a trainable dataset:

    data/dataset/yolo/
        images/{train,val,test}/...
        labels/{train,val,test}/...
        data.yaml

Splitting rule — **never split frames from one video across train/val/test**; frames in a
video are highly correlated and would leak. Stronger still: whole SITES are held out for
test, so reported accuracy reflects generalization to new locations (how the client will
actually use it). Train/val is split by video within the remaining sites.

Class list defaults to the annotation schema in docs/ANNOTATION_GUIDELINES.md.

Usage:
    python src/dataset/build_yolo_dataset.py \
        --images data/frames --labels data/annotations_yolo \
        --test-sites TON EagleCreekDrEtc --val-frac 0.15

A frame is included only if it has a label file (an empty .txt = explicit negative,
which IS included — negatives matter). Images with no .txt at all are skipped as
un-annotated.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
from collections import defaultdict

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mining"))
from filename_meta import site_from_key  # noqa: E402

DEFAULT_CLASSES = ["deer", "other_animal"]
IMG_EXTS = (".png", ".jpg", ".jpeg")


def _stable_hash(s: str) -> int:
    """Deterministic hash for reproducible val splits (avoids Python's salted hash)."""
    h = 2166136261
    for ch in s.encode():
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h


def find_images(images_root: str) -> list[str]:
    out: list[str] = []
    for ext in IMG_EXTS:
        out.extend(glob.glob(os.path.join(images_root, "**", f"*{ext}"), recursive=True))
    return sorted(set(out))


def label_path_for(img_path: str, labels_root: str, images_root: str) -> str:
    """Mirror the image's relative path under labels_root, with .txt extension."""
    rel = os.path.relpath(img_path, images_root)
    rel_txt = os.path.splitext(rel)[0] + ".txt"
    return os.path.join(labels_root, rel_txt)


def video_key_of(img_path: str) -> str:
    """Recover the video key from a frame filename like <key>_f<frame>.png.

    The key is ``SITE__transect[_vN][_SIDE]`` so the site is recoverable from it.
    """
    base = os.path.splitext(os.path.basename(img_path))[0]
    if "_f" in base:
        return base.rsplit("_f", 1)[0]
    return base


def assign_splits(images: list[str], labels_root: str, images_root: str,
                  test_sites: set[str], val_frac: float,
                  val_keys: set[str] | None = None,
                  test_keys: set[str] | None = None) -> dict[str, str]:
    """Return img_path -> split in {train,val,test}.

    Two modes:
      * Site-holdout: ``test_sites`` sends whole sites to test; remaining videos
        split train/val by ``val_keys`` (explicit) or a per-video hash to ``val_frac``.
      * Pooled site-stratified: give explicit ``test_keys`` AND ``val_keys`` (video
        lists) with no ``test_sites`` — every site can then appear in all splits.
        A video listed in test_keys wins over val_keys; everything else is train.
    """
    val_keys = val_keys or set()
    test_keys = test_keys or set()
    # Group images by video key, and keys by site.
    site_of_video: dict[str, str] = {}
    videos_with_labels: dict[str, list[str]] = defaultdict(list)
    for img in images:
        lp = label_path_for(img, labels_root, images_root)
        if not os.path.isfile(lp):
            continue  # un-annotated; skip
        key = video_key_of(img)
        site_of_video[key] = site_from_key(key)
        videos_with_labels[key].append(img)

    unknown = (val_keys | test_keys) - set(videos_with_labels)
    if unknown:
        raise SystemExit(f"--val/--test-keys not found among labelled videos: {sorted(unknown)}")

    split: dict[str, str] = {}
    for key, imgs in videos_with_labels.items():
        site = site_of_video[key]
        if key in test_keys or site in test_sites:
            dst = "test"
        elif key in val_keys:
            dst = "val"
        elif val_keys or test_keys:
            dst = "train"  # explicit-list mode: unlisted videos are train
        else:
            dst = "val" if (_stable_hash(key) % 1000) < int(val_frac * 1000) else "train"
        for im in imgs:
            split[im] = dst
    return split


def materialize(split: dict[str, str], labels_root: str, images_root: str,
                out_root: str, *, copy: bool) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for sub in ("train", "val", "test"):
        os.makedirs(os.path.join(out_root, "images", sub), exist_ok=True)
        os.makedirs(os.path.join(out_root, "labels", sub), exist_ok=True)
    for img, sub in split.items():
        lp = label_path_for(img, labels_root, images_root)
        img_out = os.path.join(out_root, "images", sub, os.path.basename(img))
        lbl_out = os.path.join(out_root, "labels", sub,
                               os.path.splitext(os.path.basename(img))[0] + ".txt")
        _place(img, img_out, copy)
        _place(lp, lbl_out, copy)
        counts[sub] += 1
    return counts


def _place(src: str, dst: str, copy: bool) -> None:
    if os.path.abspath(src) == os.path.abspath(dst):
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        if os.path.exists(dst):
            os.remove(dst)
        try:
            os.link(src, dst)  # hardlink: no extra disk, fast
        except OSError:
            shutil.copy2(src, dst)


def write_data_yaml(out_root: str, classes: list[str]) -> None:
    lines = [
        "# Auto-generated by build_yolo_dataset.py",
        f"path: {os.path.abspath(out_root)}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(classes)}",
        "names:",
    ]
    lines += [f"  {i}: {c}" for i, c in enumerate(classes)]
    with open(os.path.join(out_root, "data.yaml"), "w") as f:
        f.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", default="data/frames", help="Root of annotated images")
    p.add_argument("--labels", default="data/annotations_yolo",
                   help="Root of YOLO .txt labels (mirrors image tree)")
    p.add_argument("--out", default="data/dataset/yolo")
    p.add_argument("--test-sites", nargs="*", default=[],
                   help="Site names to hold out entirely for test")
    p.add_argument("--val-frac", type=float, default=0.15,
                   help="Fraction of non-test VIDEOS used for validation (ignored if "
                        "--val-keys is given)")
    p.add_argument("--val-keys", nargs="*", default=[],
                   help="Explicit video keys for the val split (overrides --val-frac). "
                        "Use when deer are clumped so val gets a deer-diverse sample.")
    p.add_argument("--test-keys", nargs="*", default=[],
                   help="Explicit video keys for the test split (pooled site-stratified "
                        "mode). Combine with --val-keys and no --test-sites so every "
                        "site appears in train/val/test.")
    p.add_argument("--classes", nargs="*", default=DEFAULT_CLASSES)
    p.add_argument("--copy", action="store_true",
                   help="Copy files instead of hardlinking")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    images = find_images(a.images)
    if not images:
        raise SystemExit(f"No images found under {a.images}")
    split = assign_splits(images, a.labels, a.images, set(a.test_sites), a.val_frac,
                          set(a.val_keys), set(a.test_keys))
    if not split:
        raise SystemExit(
            "No annotated images found (no matching .txt labels). "
            "Export annotations to YOLO format first."
        )
    counts = materialize(split, a.labels, a.images, a.out, copy=a.copy)
    write_data_yaml(a.out, a.classes)
    print(f"Dataset -> {a.out}")
    print(f"  classes: {a.classes}")
    print(f"  held-out test sites: {a.test_sites or '(none)'}")
    for sub in ("train", "val", "test"):
        print(f"  {sub}: {counts.get(sub, 0)} images")


if __name__ == "__main__":
    main()
