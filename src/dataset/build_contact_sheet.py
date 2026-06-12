"""
Build a single interactive HTML page to triage mined events: mark each as deer /
not-deer / unsure, then filter and export.

WHY this exists: the warm-blob miner finds *every* warm object (deer, hot rocks, tree
trunks, vehicles, people) — it cannot tell deer from non-deer, because that needs a
trained model we don't have yet. So a human makes the call here, quickly. Your verdicts:
  * are saved in the browser (localStorage) so progress survives reloads,
  * can be exported to a CSV (data/events/verdicts.csv),
  * become the clean "real deer" event set for annotation (via filter_events_by_verdict.py),
  * double as rough ground-truth deer counts per video.

Controls in the page: click a thumbnail to cycle unmarked -> deer -> not-deer -> unsure
(border color shows state); or hover a card and press D / X / U / Space. Filter buttons
show only deer / only not-deer / only unmarked. Export / Import CSV buttons at the top.
Events are sorted by score (most likely real first) so you can stop once it's all junk.

Usage:
    python src/dataset/build_contact_sheet.py
    python src/dataset/build_contact_sheet.py --min-score 0 --thumb-width 300

Reads data/events/<key>/{events.csv,thumbs/}; writes data/events/review/review.html.
"""
from __future__ import annotations

import argparse
import csv
import html
import os
from collections import defaultdict

REVIEW_DIRNAME = "review"


def _read_csv(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _score_color(score: float) -> str:
    if score >= 35:
        return "#1b7f3b"
    if score >= 27:
        return "#7f7f1b"
    return "#9c3b1b"


CSS = """
* { box-sizing: border-box; }
body { font-family: system-ui, sans-serif; margin: 0; background:#111; color:#eee; }
a { color:#7db3ff; }
header { position: sticky; top:0; z-index:10; background:#161616;
         border-bottom:1px solid #333; padding:10px 14px; }
h1 { font-size:18px; margin:0 0 8px; }
h2 { font-size:15px; margin:22px 14px 8px; color:#ffd479; scroll-margin-top:90px; }
.controls { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
button, select { background:#262626; color:#eee; border:1px solid #444; border-radius:6px;
        padding:5px 10px; font-size:13px; cursor:pointer; }
button:hover { background:#333; }
button.active { background:#2b5fa8; border-color:#3b7fd8; }
.counts span { margin-right:12px; font-size:13px; }
.dot { display:inline-block; width:10px; height:10px; border-radius:50%;
       margin-right:4px; vertical-align:middle; }
.grid { display:flex; flex-wrap:wrap; gap:10px; padding:0 14px 40px; }
.card { background:#1c1c1c; border:3px solid #333; border-radius:6px; padding:5px;
        cursor:pointer; outline:none; }
.card[data-v="deer"]   { border-color:#28c060; }
.card[data-v="no"]     { border-color:#d83b1b; opacity:.6; }
.card[data-v="unsure"] { border-color:#e0b020; }
.card img { display:block; border-radius:4px; }
.cap { font-size:11px; line-height:1.4; margin-top:4px; }
.badge { display:inline-block; padding:1px 6px; border-radius:8px; color:#fff;
         font-weight:600; }
.state { float:right; font-weight:700; }
.card[data-v="deer"]   .state::after { content:"DEER"; color:#28c060; }
.card[data-v="no"]     .state::after { content:"NO";   color:#d83b1b; }
.card[data-v="unsure"] .state::after { content:"?";    color:#e0b020; }
body.f-deer   .card:not([data-v="deer"])   { display:none; }
body.f-no     .card:not([data-v="no"])     { display:none; }
body.f-unsure .card:not([data-v="unsure"]) { display:none; }
body.f-unmarked .card[data-v]              { display:none; }
"""

JS = """
const SK='flir_verdicts_v1';
let V=JSON.parse(localStorage.getItem(SK)||'{}');
const NEXT={'':'deer','deer':'no','no':'unsure','unsure':''};
let hovered=null;
function save(){localStorage.setItem(SK,JSON.stringify(V));}
function paint(c){const v=V[c.dataset.id]; if(v)c.dataset.v=v; else c.removeAttribute('data-v');}
function setV(id,val){ if(val)V[id]=val; else delete V[id]; save();
  const c=document.querySelector(`.card[data-id="${CSS.escape(id)}"]`); if(c)paint(c); counts(); }
function cycle(id){ setV(id, NEXT[V[id]||'']); }
function counts(){ let d=0,n=0,u=0; for(const k in V){const x=V[k];
  if(x==='deer')d++; else if(x==='no')n++; else if(x==='unsure')u++;}
  const total=document.querySelectorAll('.card').length;
  document.getElementById('cD').textContent=d;
  document.getElementById('cN').textContent=n;
  document.getElementById('cU').textContent=u;
  document.getElementById('cM').textContent=total-d-n-u;
  document.getElementById('cT').textContent=total; }
function filter(mode,btn){ document.body.className='';
  document.querySelectorAll('.fbtn').forEach(b=>b.classList.remove('active'));
  if(mode){document.body.classList.add('f-'+mode); btn.classList.add('active');} }
function exportCsv(){ let rows=['video_key,event_id,verdict'];
  for(const k in V){const i=k.lastIndexOf('#'); rows.push(`${k.slice(0,i)},${k.slice(i+1)},${V[k]}`);}
  const blob=new Blob([rows.join('\\n')],{type:'text/csv'});
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob);
  a.download='verdicts.csv'; a.click(); }
function importCsv(ev){ const f=ev.target.files[0]; if(!f)return; const r=new FileReader();
  r.onload=()=>{ const lines=r.result.split(/\\r?\\n/).slice(1);
    for(const ln of lines){ if(!ln.trim())continue; const p=ln.split(',');
      if(p.length>=3) V[p[0]+'#'+p[1]]=p[2].trim(); }
    save(); document.querySelectorAll('.card').forEach(paint); counts(); };
  r.readAsText(f); }
document.addEventListener('DOMContentLoaded',()=>{
  document.querySelectorAll('.card').forEach(c=>{ paint(c);
    c.addEventListener('click',()=>cycle(c.dataset.id));
    c.addEventListener('mouseenter',()=>hovered=c.dataset.id);
    c.addEventListener('mouseleave',()=>{if(hovered===c.dataset.id)hovered=null;}); });
  counts(); });
document.addEventListener('keydown',e=>{ if(!hovered)return; const k=e.key.toLowerCase();
  if(k==='d')setV(hovered,'deer'); else if(k==='x')setV(hovered,'no');
  else if(k==='u')setV(hovered,'unsure'); else if(k===' '){e.preventDefault();cycle(hovered);} });
"""


def build_page(summary: list[dict], events_root: str, review_dir: str,
               thumb_width: int, min_score: float) -> int:
    by_site: dict[str, list[dict]] = defaultdict(list)
    for r in summary:
        by_site[r.get("site", "?")].append(r)

    sections, total_shown = [], 0
    nav_links = []
    for site in sorted(by_site):
        vids = sorted(by_site[site], key=lambda r: -int(r.get("n_events", 0) or 0))
        for r in vids:
            key = r["key"]
            events = _read_csv(os.path.join(events_root, key, "events.csv"))
            events = [e for e in events
                      if float(e.get("peak_score", 0) or 0) >= min_score]
            events.sort(key=lambda e: -float(e.get("peak_score", 0) or 0))
            cards = []
            for e in events:
                eid = int(e["event_id"])
                thumb = os.path.join(events_root, key, "thumbs", f"event_{eid:03d}.png")
                if not os.path.isfile(thumb):
                    continue
                rel = os.path.relpath(thumb, review_dir).replace("\\", "/")
                score = float(e.get("peak_score", 0) or 0)
                gid = f"{key}#{eid}"
                cap = (f'<span class="state"></span>e{eid} &nbsp;'
                       f'{e["start_s"]}&ndash;{e["end_s"]}s<br>'
                       f'<span class="badge" style="background:{_score_color(score)}">'
                       f'score {score:.0f}</span> &nbsp;{e.get("n_hits","?")} hits')
                cards.append(
                    f'<div class="card" tabindex="0" data-id="{html.escape(gid)}">'
                    f'<img loading="lazy" src="{rel}" width="{thumb_width}">'
                    f'<div class="cap">{cap}</div></div>'
                )
                total_shown += 1
            if cards:
                anchor = html.escape(key)
                nav_links.append(f'<option value="{anchor}">{anchor} ({len(cards)})</option>')
                sections.append(
                    f'<h2 id="{anchor}">{html.escape(site)} / {html.escape(key)} '
                    f'&mdash; {len(cards)} events</h2>'
                    f'<div class="grid">{"".join(cards)}</div>'
                )

    nav = ('<select onchange="if(this.value)location.hash=this.value">'
           '<option value="">jump to video…</option>' + "".join(nav_links) + '</select>')
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>FLIR event triage</title><style>{CSS}</style></head><body>
<header>
  <h1>FLIR event triage &mdash; click a thumbnail to mark; hover + D / X / U / Space</h1>
  <div class="controls">
    <span class="counts">
      <span><span class="dot" style="background:#28c060"></span>deer <b id="cD">0</b></span>
      <span><span class="dot" style="background:#d83b1b"></span>not <b id="cN">0</b></span>
      <span><span class="dot" style="background:#e0b020"></span>unsure <b id="cU">0</b></span>
      <span>unmarked <b id="cM">0</b> / <b id="cT">0</b></span>
    </span>
    <button class="fbtn active" onclick="filter('',this)">all</button>
    <button class="fbtn" onclick="filter('deer',this)">deer</button>
    <button class="fbtn" onclick="filter('no',this)">not-deer</button>
    <button class="fbtn" onclick="filter('unsure',this)">unsure</button>
    <button class="fbtn" onclick="filter('unmarked',this)">unmarked</button>
    {nav}
    <button onclick="exportCsv()">⬇ export CSV</button>
    <label style="cursor:pointer">⬆ import CSV
      <input type="file" accept=".csv" style="display:none" onchange="importCsv(event)">
    </label>
  </div>
</header>
{''.join(sections)}
<script>{JS}</script>
</body></html>"""
    out = os.path.join(review_dir, "review.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    return total_shown


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
    n = build_page(summary, a.events_root, review_dir, a.thumb_width, a.min_score)
    out = os.path.join(review_dir, "review.html")
    print(f"Triage page -> {out}")
    print(f"{n} event thumbnails. Open it, mark deer/not-deer, then 'export CSV'.")


if __name__ == "__main__":
    main()
