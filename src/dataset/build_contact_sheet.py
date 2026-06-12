"""
Build an HTML contact sheet to visually review mined events (verify detection quality).

For each mined video it lays out the per-event peak-frame thumbnails (with the detection
boxes already drawn) in a grid, captioned with event id, time range, peak score, and hit
count. A top-level ``index.html`` links every video, grouped by site, with counts.

Open ``data/events/review/index.html`` in a browser and skim: are the red boxes on real
deer, or on hot rocks / motion streaks / vehicles? Use the score coloring (low scores
tend to be junk) to judge whether to raise the miner's --min-contrast threshold.

Usage:
    python src/dataset/build_contact_sheet.py
    python src/dataset/build_contact_sheet.py --min-score 0 --thumb-width 320

Reads data/events/<key>/{events.csv,thumbs/}; writes data/events/review/*.html.
Thumbnails are referenced in place (not copied), so this is fast and tiny.
"""
from __future__ import annotations

import argparse
import csv
import html
import os

REVIEW_DIRNAME = "review"


def _read_csv(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _score_color(score: float) -> str:
    """Green (high contrast, likely real) -> red (low, likely junk)."""
    if score >= 35:
        return "#1b7f3b"
    if score >= 27:
        return "#7f7f1b"
    return "#9c3b1b"


def _page_css() -> str:
    return """
    body { font-family: system-ui, sans-serif; margin: 16px; background:#111; color:#eee; }
    a { color:#7db3ff; }
    h1 { font-size: 20px; } h2 { font-size: 16px; margin-top: 28px; }
    .grid { display: flex; flex-wrap: wrap; gap: 10px; }
    .card { background:#1c1c1c; border:1px solid #333; border-radius:6px; padding:6px; }
    .card img { display:block; border-radius:4px; }
    .cap { font-size: 11px; line-height:1.4; margin-top:4px; }
    .badge { display:inline-block; padding:1px 6px; border-radius:8px; color:#fff;
             font-weight:600; }
    .controls { position: sticky; top:0; background:#111; padding:8px 0; }
    .site { color:#ffd479; }
    table { border-collapse: collapse; } td, th { padding:4px 10px; text-align:left;
            border-bottom:1px solid #333; font-size:13px; }
    """


def build_video_page(key: str, events: list[dict], events_root: str,
                     review_dir: str, thumb_width: int, min_score: float) -> int:
    rows = []
    shown = 0
    for e in events:
        score = float(e.get("peak_score", 0) or 0)
        if score < min_score:
            continue
        eid = int(e["event_id"])
        thumb = os.path.join(events_root, key, "thumbs", f"event_{eid:03d}.png")
        if not os.path.isfile(thumb):
            continue
        # Relative path from review_dir to the thumbnail.
        rel = os.path.relpath(thumb, review_dir).replace("\\", "/")
        cap = (f'e{eid} &nbsp; {e["start_s"]}&ndash;{e["end_s"]}s<br>'
               f'<span class="badge" style="background:{_score_color(score)}">'
               f'score {score:.0f}</span> &nbsp; {e.get("n_hits","?")} hits')
        rows.append(
            f'<div class="card"><a href="{rel}" target="_blank">'
            f'<img src="{rel}" width="{thumb_width}"></a>'
            f'<div class="cap">{cap}</div></div>'
        )
        shown += 1

    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{html.escape(key)}</title><style>{_page_css()}</style></head><body>
<p><a href="index.html">&larr; all videos</a></p>
<h1>{html.escape(key)}</h1>
<p>{shown} events shown (min score {min_score:g}). Click a frame for full size.</p>
<div class="grid">{''.join(rows)}</div>
</body></html>"""
    with open(os.path.join(review_dir, f"{key}.html"), "w", encoding="utf-8") as f:
        f.write(page)
    return shown


def build_index(summary: list[dict], review_dir: str) -> None:
    from collections import defaultdict
    by_site: dict[str, list[dict]] = defaultdict(list)
    for r in summary:
        by_site[r.get("site", "?")].append(r)

    sections = []
    for site in sorted(by_site):
        vids = sorted(by_site[site], key=lambda r: -int(r.get("n_events", 0) or 0))
        trows = "".join(
            f'<tr><td><a href="{html.escape(r["key"])}.html">'
            f'{html.escape(r["key"])}</a></td>'
            f'<td>{r.get("n_events","?")}</td>'
            f'<td>{r.get("event_seconds","?")}</td></tr>'
            for r in vids
        )
        n_evt = sum(int(r.get("n_events", 0) or 0) for r in vids)
        sections.append(
            f'<h2 class="site">{html.escape(site)} '
            f'&mdash; {len(vids)} videos, {n_evt} events</h2>'
            f'<table><tr><th>video</th><th>events</th><th>event seconds</th></tr>'
            f'{trows}</table>'
        )

    total_v = len(summary)
    total_e = sum(int(r.get("n_events", 0) or 0) for r in summary)
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Event review</title><style>{_page_css()}</style></head><body>
<h1>FLIR event review &mdash; {total_v} videos, {total_e} events</h1>
<p>Click a video to see its event thumbnails. Verify the red boxes are on real deer.</p>
{''.join(sections)}
</body></html>"""
    with open(os.path.join(review_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--events-root", default="data/events")
    p.add_argument("--thumb-width", type=int, default=300)
    p.add_argument("--min-score", type=float, default=0.0,
                   help="Hide events below this peak score")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    review_dir = os.path.join(a.events_root, REVIEW_DIRNAME)
    os.makedirs(review_dir, exist_ok=True)

    summary = _read_csv(os.path.join(a.events_root, "summary_by_video.csv"))
    if not summary:
        raise SystemExit("No summary_by_video.csv — run batch_mine.py first.")

    total_shown = 0
    for r in summary:
        key = r["key"]
        events = _read_csv(os.path.join(a.events_root, key, "events.csv"))
        total_shown += build_video_page(key, events, a.events_root, review_dir,
                                        a.thumb_width, a.min_score)
    build_index(summary, review_dir)
    print(f"Contact sheet -> {os.path.join(review_dir, 'index.html')}")
    print(f"{total_shown} event thumbnails across {len(summary)} videos.")


if __name__ == "__main__":
    main()
