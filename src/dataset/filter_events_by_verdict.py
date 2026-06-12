"""
Filter the mined events down to human-confirmed deer, using verdicts.csv from the
triage page (build_contact_sheet.py -> "export CSV").

Produces:
  * data/events/master_events_deer.csv  — only events you marked 'deer' (optionally
    also 'unsure'), carrying all the original metadata. This feeds
    select_for_annotation.py so you only ever cut clips for real deer.
  * data/events/verdict_summary.csv      — per-video deer/not/unsure/unmarked tallies =
    rough ground-truth deer-event counts per video.

Usage:
    python src/dataset/filter_events_by_verdict.py
    python src/dataset/filter_events_by_verdict.py --include-unsure
    python src/dataset/filter_events_by_verdict.py --verdicts path/to/verdicts.csv

Then:
    python src/dataset/select_for_annotation.py --master data/events/master_events_deer.csv
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict


def _read_csv(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_verdicts(path: str) -> dict[tuple[str, str], str]:
    """(video_key, event_id) -> verdict."""
    out: dict[tuple[str, str], str] = {}
    for r in _read_csv(path):
        key = r.get("video_key") or r.get("key")
        eid = r.get("event_id")
        v = (r.get("verdict") or "").strip().lower()
        if key is not None and eid is not None and v:
            out[(key, str(eid))] = v
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--master", default="data/events/master_events.csv")
    p.add_argument("--verdicts", default="data/events/verdicts.csv")
    p.add_argument("--out", default="data/events/master_events_deer.csv")
    p.add_argument("--summary", default="data/events/verdict_summary.csv")
    p.add_argument("--include-unsure", action="store_true",
                   help="Also keep events marked 'unsure' (for a second review pass)")
    a = p.parse_args()

    master = _read_csv(a.master)
    if not master:
        raise SystemExit(f"No master events at {a.master}. Run batch_mine.py first.")
    verdicts = load_verdicts(a.verdicts)
    if not verdicts:
        raise SystemExit(
            f"No verdicts at {a.verdicts}. Open the triage page, mark events, "
            f"click 'export CSV', and save it there."
        )

    keep_set = {"deer"} | ({"unsure"} if a.include_unsure else set())
    kept = []
    tally: dict[str, dict[str, int]] = defaultdict(
        lambda: {"deer": 0, "no": 0, "unsure": 0, "unmarked": 0})

    for r in master:
        key = r.get("key")
        eid = str(r.get("event_id"))
        v = verdicts.get((key, eid), "unmarked")
        tally[key][v if v in ("deer", "no", "unsure") else "unmarked"] += 1
        if v in keep_set:
            kept.append(r)

    if kept:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(master[0].keys()))
            w.writeheader()
            w.writerows(kept)

    # Per-video verdict summary (rough ground-truth deer counts).
    site_of = {r["key"]: r.get("site", "?") for r in master}
    with open(a.summary, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "site", "deer", "not_deer", "unsure", "unmarked"])
        for key in sorted(tally):
            t = tally[key]
            w.writerow([key, site_of.get(key, "?"), t["deer"], t["no"],
                        t["unsure"], t["unmarked"]])

    n_deer = sum(t["deer"] for t in tally.values())
    n_marked = sum(t["deer"] + t["no"] + t["unsure"] for t in tally.values())
    print(f"Verdicts loaded: {len(verdicts)}  ({n_marked}/{len(master)} events marked)")
    print(f"Kept {len(kept)} events ({'deer+unsure' if a.include_unsure else 'deer'}) "
          f"-> {a.out}")
    print(f"Total marked deer: {n_deer}. Per-video summary -> {a.summary}")


if __name__ == "__main__":
    main()
