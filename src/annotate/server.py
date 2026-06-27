"""
Local box-correction web app for the prepared frames. Zero dependencies (Python stdlib).

Run:
    python src/annotate/prepare_frames.py      # once, to extract frames + seed boxes
    python src/annotate/server.py              # then open http://localhost:8000

In the browser you correct each frame:
  * delete wrong boxes (hot rocks, tree trunks, stray fragments),
  * fix a partial box so it covers the WHOLE deer,
  * merge 3 fragment-boxes on one deer into a single box (delete two, resize one),
  * draw any deer the detector missed,
  * a frame with no deer: click "No deer" to clear it (valid negative example).

Boxes are saved as YOLO labels in data/annotate/labels/ (class 0 = deer), so they plug
straight into src/dataset/build_yolo_dataset.py. Progress (which frames you finished) is
tracked in data/annotate/status.json.

Controls: drag empty area = draw box; drag box = move; drag corner = resize; click box
then Delete = remove; ← / → = prev/next (auto-saves); N = no-deer clear; S = save.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import posixpath
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = "data/annotate"
FRAMES_DIR = os.path.join(ROOT, "frames")
LABELS_DIR = os.path.join(ROOT, "labels")
STATUS_PATH = os.path.join(ROOT, "status.json")
INDEX_NAME = "frames.csv"


def _set_root(root: str) -> None:
    """Point the app at any annotate-style dir (frames/, labels/, <index>.csv)."""
    global ROOT, FRAMES_DIR, LABELS_DIR, STATUS_PATH
    ROOT = root
    FRAMES_DIR = os.path.join(ROOT, "frames")
    LABELS_DIR = os.path.join(ROOT, "labels")
    STATUS_PATH = os.path.join(ROOT, "status.json")


def _load_index() -> list[dict]:
    path = os.path.join(ROOT, INDEX_NAME)
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _load_status() -> set:
    if os.path.isfile(STATUS_PATH):
        try:
            return set(json.load(open(STATUS_PATH)).get("done", []))
        except Exception:
            return set()
    return set()


def _save_status(done: set) -> None:
    json.dump({"done": sorted(done)}, open(STATUS_PATH, "w"))


def _read_label(name: str) -> list[list[float]]:
    stem = os.path.splitext(name)[0]
    path = os.path.join(LABELS_DIR, stem + ".txt")
    boxes = []
    if os.path.isfile(path):
        for line in open(path):
            parts = line.split()
            if len(parts) == 5:
                boxes.append([float(x) for x in parts])
    return boxes


def _write_label(name: str, boxes: list[list[float]]) -> None:
    stem = os.path.splitext(name)[0]
    os.makedirs(LABELS_DIR, exist_ok=True)
    with open(os.path.join(LABELS_DIR, stem + ".txt"), "w") as f:
        for b in boxes:
            cls = int(b[0])
            f.write(f"{cls} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")


class Handler(BaseHTTPRequestHandler):
    def _send(self, code=200, body=b"", ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def log_message(self, *a):
        pass  # quiet

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            self._send(200, INDEX_HTML.encode(), "text/html; charset=utf-8")
        elif u.path == "/api/frames":
            done = _load_status()
            idx = _load_index()
            for r in idx:
                r["done"] = r["name"] in done
            self._json({"frames": idx, "n_done": len(done)})
        elif u.path == "/api/label":
            name = parse_qs(u.query).get("name", [""])[0]
            self._json({"name": name, "boxes": _read_label(name)})
        elif u.path.startswith("/frames/"):
            self._serve_image(u.path[len("/frames/"):])
        else:
            self._send(404, b"not found", "text/plain")

    def _serve_image(self, name):
        safe = posixpath.normpath(name).lstrip("/")
        path = os.path.join(FRAMES_DIR, safe)
        if not os.path.isfile(path):
            self._send(404, b"no image", "text/plain")
            return
        with open(path, "rb") as f:
            self._send(200, f.read(), "image/png")

    def do_POST(self):
        u = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(length) or b"{}")
        if u.path == "/api/label":
            name = data.get("name")
            boxes = data.get("boxes", [])
            if not name:
                self._json({"error": "no name"}, 400)
                return
            _write_label(name, boxes)
            done = _load_status()
            if data.get("done"):
                done.add(name)
            else:
                done.discard(name)
            _save_status(done)
            self._json({"ok": True, "n_done": len(done)})
        else:
            self._send(404, b"not found", "text/plain")


INDEX_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>Deer box correction</title>
<style>
* { box-sizing:border-box; }
body { margin:0; font-family:system-ui,sans-serif; background:#111; color:#eee;
       display:flex; height:100vh; overflow:hidden; }
#side { width:240px; background:#181818; border-right:1px solid #333; overflow-y:auto;
        flex-shrink:0; }
#side h3 { margin:10px; font-size:13px; color:#ffd479; }
.fitem { padding:6px 10px; font-size:12px; cursor:pointer; border-bottom:1px solid #222;
         display:flex; justify-content:space-between; }
.fitem:hover { background:#262626; } .fitem.cur { background:#2b5fa8; }
.fitem.done span.k { color:#28c060; }
#main { flex:1; display:flex; flex-direction:column; overflow:hidden; }
#bar { padding:8px 12px; background:#161616; border-bottom:1px solid #333;
       display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
button { background:#262626; color:#eee; border:1px solid #444; border-radius:6px;
         padding:6px 12px; font-size:13px; cursor:pointer; } button:hover{background:#333;}
button.primary { background:#2b5fa8; border-color:#3b7fd8; }
#stagewrap { flex:1; overflow:auto; padding:16px; }
#stage { position:relative; cursor:crosshair; user-select:none; touch-action:none;
         box-shadow:0 0 0 1px #333; }
#stage img { display:block; pointer-events:none; }
.box { position:absolute; border:2px solid #28c060; background:rgba(40,192,96,.12); }
.box.sel { border-color:#ffd479; background:rgba(255,212,121,.18); }
.box .h { position:absolute; right:-6px; bottom:-6px; width:12px; height:12px;
          background:#ffd479; border-radius:2px; cursor:nwse-resize; }
.box .x { position:absolute; left:-2px; top:-18px; background:#d83b1b; color:#fff;
          font-size:11px; padding:0 4px; border-radius:3px; cursor:pointer; }
#hint { font-size:11px; color:#999; margin-left:auto; }
.pill { font-size:12px; color:#bbb; }
</style></head><body>
<div id="side"><h3 id="prog">…</h3><div id="list"></div></div>
<div id="main">
  <div id="bar">
    <button onclick="go(-1)">← Prev</button>
    <button onclick="go(1)">Next →</button>
    <button class="primary" onclick="save(true)">Save ✓ (S)</button>
    <button onclick="clearBoxes()">No deer (N)</button>
    <button onclick="zoom(-0.25)">−</button><button onclick="zoom(0.25)">+</button>
    <span class="pill" id="meta"></span>
    <span id="hint">drag empty=draw · drag box=move · corner=resize · click+Delete=remove</span>
  </div>
  <div id="stagewrap"><div id="stage"></div></div>
</div>
<script>
const W=640,H=512; let SCALE=1.8;
let frames=[], cur=-1, boxes=[], sel=-1, dirty=false;
const stage=document.getElementById('stage');
const img=new Image(); stage.appendChild(img);

async function load(){ const r=await fetch('/api/frames'); const d=await r.json();
  frames=d.frames; renderList(); if(frames.length) open(0); updateProg(d.n_done); }
function updateProg(n){ document.getElementById('prog').textContent=
  `${n||0} done / ${frames.length} frames`; }
function renderList(){ const el=document.getElementById('list'); el.innerHTML='';
  frames.forEach((f,i)=>{ const d=document.createElement('div');
    d.className='fitem'+(i===cur?' cur':'')+(f.done?' done':'');
    d.innerHTML=`<span class="k">${f.key}</span><span>${f.score}</span>`;
    d.title=f.name; d.onclick=()=>{ if(dirty)save(false); open(i); }; el.appendChild(d); }); }
async function open(i){ if(i<0||i>=frames.length)return; cur=i; sel=-1; dirty=false;
  const f=frames[i];
  document.getElementById('meta').textContent=`${f.key}  ·  frame ${f.src_frame}  ·  score ${f.score}  ·  [${i+1}/${frames.length}]`;
  const r=await fetch('/api/label?name='+encodeURIComponent(f.name)); const d=await r.json();
  boxes=d.boxes.map(b=>norm2px(b));
  img.src='/frames/'+encodeURIComponent(f.name)+'?t='+Date.now();
  img.onload=()=>{ applyScale(); }; renderList(); }
function applyScale(){ stage.style.width=(W*SCALE)+'px'; stage.style.height=(H*SCALE)+'px';
  img.style.width=(W*SCALE)+'px'; img.style.height=(H*SCALE)+'px'; draw(); }
function zoom(d){ SCALE=Math.max(1,Math.min(4,SCALE+d)); // rescale existing px boxes
  const k=(SCALE)/(SCALE-d); boxes.forEach(b=>{b.x*=k;b.y*=k;b.w*=k;b.h*=k;}); applyScale(); }
function norm2px(b){ return {cls:b[0], x:(b[1]-b[3]/2)*W*SCALE, y:(b[2]-b[4]/2)*H*SCALE,
  w:b[3]*W*SCALE, h:b[4]*H*SCALE}; }
function px2norm(b){ const xc=(b.x+b.w/2)/(W*SCALE), yc=(b.y+b.h/2)/(H*SCALE);
  return [b.cls||0, xc, yc, b.w/(W*SCALE), b.h/(H*SCALE)]; }
function draw(){ [...stage.querySelectorAll('.box')].forEach(e=>e.remove());
  boxes.forEach((b,i)=>{ const d=document.createElement('div');
    d.className='box'+(i===sel?' sel':''); d.style.left=b.x+'px'; d.style.top=b.y+'px';
    d.style.width=b.w+'px'; d.style.height=b.h+'px'; d.dataset.i=i;
    const x=document.createElement('div'); x.className='x'; x.textContent='×';
    x.onclick=(e)=>{e.stopPropagation(); boxes.splice(i,1); sel=-1; dirty=true; draw();};
    const h=document.createElement('div'); h.className='h'; h.dataset.resize=i;
    d.appendChild(x); d.appendChild(h); stage.appendChild(d); }); }

let mode=null, start=null, oi=-1;
stage.addEventListener('pointerdown',e=>{ const rect=stage.getBoundingClientRect();
  const mx=e.clientX-rect.left, my=e.clientY-rect.top;
  if(e.target.dataset.resize!==undefined){ mode='resize'; oi=+e.target.dataset.resize;
    sel=oi; start=[mx,my]; draw(); return; }
  const boxEl=e.target.closest('.box');
  if(boxEl){ oi=+boxEl.dataset.i; sel=oi; mode='move';
    start=[mx-boxes[oi].x, my-boxes[oi].y]; draw(); return; }
  mode='draw'; start=[mx,my]; boxes.push({cls:0,x:mx,y:my,w:0,h:0}); oi=boxes.length-1;
  sel=oi; stage.setPointerCapture(e.pointerId); });
stage.addEventListener('pointermove',e=>{ if(!mode)return; const rect=stage.getBoundingClientRect();
  const mx=e.clientX-rect.left, my=e.clientY-rect.top; const b=boxes[oi]; if(!b)return;
  if(mode==='draw'){ b.x=Math.min(start[0],mx); b.y=Math.min(start[1],my);
    b.w=Math.abs(mx-start[0]); b.h=Math.abs(my-start[1]); }
  else if(mode==='move'){ b.x=mx-start[0]; b.y=my-start[1]; }
  else if(mode==='resize'){ b.w=Math.max(4,mx-b.x); b.h=Math.max(4,my-b.y); }
  dirty=true; draw(); });
stage.addEventListener('pointerup',e=>{ if(mode==='draw'){ const b=boxes[oi];
    if(b.w<5||b.h<5) boxes.splice(oi,1); } mode=null; oi=-1; draw(); });

function clearBoxes(){ boxes=[]; sel=-1; dirty=true; draw(); }
async function save(done){ if(cur<0)return; const f=frames[cur];
  const payload={name:f.name, boxes:boxes.map(px2norm), done:!!done};
  const r=await fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(payload)}); const d=await r.json();
  dirty=false; if(done){ frames[cur].done=true; renderList(); updateProg(d.n_done);
    if(cur<frames.length-1) open(cur+1); } }
function go(step){ if(dirty)save(false); open(cur+step); }
document.addEventListener('keydown',e=>{ if(e.target.tagName==='INPUT')return;
  if(e.key==='ArrowRight')go(1); else if(e.key==='ArrowLeft')go(-1);
  else if(e.key==='s'||e.key==='S')save(true);
  else if(e.key==='n'||e.key==='N')clearBoxes();
  else if(e.key==='Delete'||e.key==='Backspace'){ if(sel>=0){ boxes.splice(sel,1);
    sel=-1; dirty=true; draw(); e.preventDefault(); } } });
load();
</script></body></html>"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--root", default="data/annotate",
                   help="annotate-style dir with frames/, labels/, and an index csv "
                        "(e.g. data/annotate/harvest to verify harvested pre-labels)")
    p.add_argument("--index", default="frames.csv",
                   help="index csv filename inside --root")
    a = p.parse_args()
    global INDEX_NAME
    INDEX_NAME = a.index
    _set_root(a.root)
    if not _load_index():
        raise SystemExit(f"No {os.path.join(a.root, a.index)} — "
                         "run the harvest/prepare step first.")
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"Box-correction app at http://localhost:{a.port}  (Ctrl+C to stop)")
    print(f"{len(_load_index())} frames, {len(_load_status())} marked done.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
