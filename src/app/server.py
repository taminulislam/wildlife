#!/usr/bin/env python3
"""
TRACT web interface. Upload a raw thermal video, get it back annotated with detections,
track identities and a running count.

Deliberately dependency-free: Python's own http.server, no Gradio, Streamlit or Flask.
The cluster environment has none of them, the project is near its inode quota, and a
reviewer or collaborator should be able to run this from a fresh checkout of the repo
with nothing but the inference environment already needed to run the pipeline.

  python src/app/server.py --port 8080 --device 0

On a compute node, forward the port from your laptop:
  ssh -L 8080:<node>:8080 <user>@delta.ncsa.illinois.edu
then open http://localhost:8080
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_APPDIR = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.abspath(os.path.join(_APPDIR, "..", ".."))
sys.path.insert(0, _APPDIR)
import engine                                                          # noqa: E402

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
UPLOADS: dict[str, dict] = {}
CHUNK_LIMIT = 32 * 1024 ** 2      # refuse a single chunk larger than this
CFG = engine.Config()
WORK = "/tmp/tract_app"
MAX_BYTES = 4 * 1024 ** 3          # 4 GB upload ceiling
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".m4v")
# Roots the server will read from when given a path instead of an upload. Anything outside
# these is refused, so the browser cannot be used to read arbitrary files off the cluster.
BROWSE_ROOTS: list[str] = []

PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TRACT — thermal deer counting</title><style>
:root{--bg:#12151a;--panel:#1a1f27;--line:#2a323d;--ink:#e8ecf1;--dim:#9aa7b4;
--accent:#5C7FA0;--accent2:#A7BFD4;--warm:#C58164;--ok:#3fa66f}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:22px 28px;border-bottom:1px solid var(--line);display:flex;
align-items:baseline;gap:14px;flex-wrap:wrap}
h1{margin:0;font-size:20px;letter-spacing:.2px}
header .sub{color:var(--dim);font-size:13px}
main{max-width:1000px;margin:0 auto;padding:26px 20px 60px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
padding:20px;margin-bottom:18px}
.drop{border:1.5px dashed var(--line);border-radius:10px;padding:34px;text-align:center;
color:var(--dim);cursor:pointer;transition:.15s}
.drop:hover,.drop.hot{border-color:var(--accent2);color:var(--ink);background:#1e242d}
.row{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-end}
label{display:block;font-size:12px;color:var(--dim);margin-bottom:5px}
input[type=number],select{background:#11151b;color:var(--ink);border:1px solid var(--line);
border-radius:7px;padding:8px 10px;width:118px;font:inherit}
button{background:var(--accent);color:#fff;border:0;border-radius:8px;padding:11px 22px;
font:inherit;font-weight:600;cursor:pointer}
button:disabled{opacity:.45;cursor:default}
button.ghost{background:transparent;border:1px solid var(--line);color:var(--ink);font-weight:500}
.bar{height:7px;background:#11151b;border-radius:4px;overflow:hidden;margin:12px 0 8px}
.bar>i{display:block;height:100%;width:0;background:var(--accent2);transition:width .25s}
.stat{display:flex;gap:26px;flex-wrap:wrap;margin:6px 0 2px}
.stat div b{display:block;font-size:26px;line-height:1.2}
.stat div span{font-size:12px;color:var(--dim)}
video{width:100%;border-radius:9px;background:#000;margin-top:6px}
.muted{color:var(--dim);font-size:13px}
.err{color:#e08a7a;white-space:pre-wrap;font-family:ui-monospace,monospace;font-size:12.5px}
a.dl{color:var(--accent2);margin-right:18px}
code{background:#11151b;padding:1px 5px;border-radius:4px;font-size:13px}
.tabs{display:flex;gap:8px;margin-bottom:14px}
.tab{background:transparent;border:1px solid var(--line);color:var(--dim);font-weight:500;
padding:7px 14px;border-radius:7px}
.tab.on{background:#222a34;color:var(--ink);border-color:var(--accent)}
.crumb{font-family:ui-monospace,monospace;font-size:12.5px;color:var(--dim);
margin-bottom:8px;word-break:break-all}
.browse{max-height:270px;overflow:auto;border:1px solid var(--line);border-radius:8px}
.it{display:flex;justify-content:space-between;gap:12px;padding:8px 12px;cursor:pointer;
border-bottom:1px solid #222a34;font-size:14px}
.it:last-child{border-bottom:0}.it:hover{background:#222a34}
.it .sz{color:var(--dim);font-size:12px;white-space:nowrap}
.it.dir b{color:var(--accent2);font-weight:500}
.chosen{margin-top:10px;color:var(--accent2);font-family:ui-monospace,monospace;font-size:12.5px;
word-break:break-all}
</style></head><body>
<header><h1>TRACT</h1>
<span class="sub">Thermal Road-transect Animal Counting and Tracking — detect, track, count</span></header>
<main>
  <div class="card" id="upcard">
    <div class="tabs">
      <button class="tab on" id="tabPick">Choose on the cluster</button>
      <button class="tab" id="tabUp">Upload from this computer</button>
    </div>

    <div id="paneP">
      <div class="crumb" id="crumb"></div>
      <div class="browse" id="browse"></div>
      <div id="chosen" class="chosen"></div>
    </div>

    <div id="paneU" style="display:none">
      <div class="drop" id="drop">
        <div style="font-size:15px;color:var(--ink)"><b>Drop a video here</b> or click to choose</div>
        <div style="margin-top:6px">MP4, AVI or MOV &middot; up to 4&nbsp;GB</div>
        <div id="picked" style="margin-top:10px;color:var(--accent2)"></div>
      </div>
      <p class="muted" style="margin:10px 0 0">Uploading sends the whole file through the SSH
        tunnel. For a video already on Delta, the first tab is far faster and cannot time out.</p>
    </div>
    <input id="file" type="file" accept="video/*" style="display:none">
    <div class="row" style="margin-top:18px">
      <div><label>Detector confidence</label><input id="conf" type="number" step="0.01" min="0.01" max="0.9" value="0.10"></div>
      <div><label>Min frames <span title="confirmation rule">(n)</span></label><input id="nf" type="number" min="1" max="500" value="20"></div>
      <div><label>Min track score</label><input id="tc" type="number" step="0.05" min="0" max="1" value="0.65"></div>
      <div><label>Draw candidates</label>
        <select id="dc"><option value="1" selected>yes</option><option value="0">no</option></select></div>
      <div style="flex:1"></div>
      <button id="go" disabled>Run pipeline</button>
    </div>
    <p class="muted" style="margin:14px 0 0">Defaults are the published operating point.
      Lowering the track score recovers animals and admits duplicates; the paper's
      analysis of that trade is in §4.6.</p>
  </div>

  <div class="card" id="prog" style="display:none">
    <div id="stage">Starting…</div>
    <div class="bar"><i id="fill"></i></div>
    <div class="muted" id="detail"></div>
  </div>

  <div class="card" id="res" style="display:none">
    <div class="stat">
      <div><b id="count">–</b><span>deer counted</span></div>
      <div><b id="cand">–</b><span>candidate tracks</span></div>
      <div><b id="frames">–</b><span>frames</span></div>
      <div><b id="secs">–</b><span>seconds elapsed</span></div>
    </div>
    <video id="vid" controls></video>
    <p style="margin:14px 0 0">
      <a class="dl" id="dlv" download>Download annotated video</a>
      <a class="dl" id="dlc" download>Download per-track CSV</a>
      <button class="ghost" id="again">Run another</button></p>
    <p class="muted" id="rule"></p>
  </div>

  <div class="card" id="errc" style="display:none"><b>Failed</b><div class="err" id="err"></div></div>
</main>
<script>
const $=id=>document.getElementById(id);
let file=null, job=null, t0=0, mode='pick', chosen=null;
$('tabPick').onclick=()=>setMode('pick');
$('tabUp').onclick=()=>setMode('up');
function setMode(m){mode=m;
  $('tabPick').classList.toggle('on',m==='pick');$('tabUp').classList.toggle('on',m==='up');
  $('paneP').style.display=m==='pick'?'':'none';$('paneU').style.display=m==='up'?'':'none';
  $('go').disabled = m==='pick' ? !chosen : !file;}
async function browse(dir){
  const r=await fetch('/api/list?dir='+encodeURIComponent(dir||''));const j=await r.json();
  const box=$('browse');box.innerHTML='';
  if(j.error){box.innerHTML='<div class="it"><b>'+j.error+'</b></div>';return}
  $('crumb').textContent=j.dir;
  if(j.parent){box.appendChild(row('..',null,()=>browse(j.parent),true))}
  j.dirs.forEach(d=>box.appendChild(row(d+'/',null,()=>browse(j.dir+'/'+d),true)));
  j.videos.forEach(v=>box.appendChild(row(v.name,(v.size/1048576).toFixed(1)+' MB',
      ()=>{chosen=j.dir+'/'+v.name;$('chosen').textContent='selected: '+chosen;$('go').disabled=false})));
  if(!j.dirs.length&&!j.videos.length)box.innerHTML+='<div class="it"><span class="sz">empty</span></div>';
}
function row(name,size,fn,isdir){
  const d=document.createElement('div');d.className='it'+(isdir?' dir':'');
  d.innerHTML='<b>'+name.replace(/</g,'&lt;')+'</b>'+(size?'<span class="sz">'+size+'</span>':'');
  d.onclick=fn;return d;
}
browse('');
$('drop').onclick=()=>$('file').click();
$('drop').ondragover=e=>{e.preventDefault();$('drop').classList.add('hot')};
$('drop').ondragleave=()=>$('drop').classList.remove('hot');
$('drop').ondrop=e=>{e.preventDefault();$('drop').classList.remove('hot');pick(e.dataTransfer.files[0])};
$('file').onchange=e=>pick(e.target.files[0]);
function pick(f){if(!f)return;file=f;$('picked').textContent=f.name+'  ('+(f.size/1048576).toFixed(1)+' MB)';$('go').disabled=false}
$('again').onclick=()=>{$('res').style.display='none';$('upcard').style.display='';
  file=null;chosen=null;$('picked').textContent='';$('chosen').textContent='';$('go').disabled=true;browse('')};
$('go').onclick=async()=>{
  if(mode==='pick'?!chosen:!file)return;
  $('go').disabled=true;$('errc').style.display='none';$('prog').style.display='';$('res').style.display='none';
  const body={conf:$('conf').value,min_frames:$('nf').value,
              min_topk:$('tc').value,draw_candidates:$('dc').value};
  t0=Date.now();
  try{
    if(mode==='pick'){ body.path=chosen; }
    else{ body.upload_id=await uploadChunked(file); }
    const r=await fetch('/api/run',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const j=await r.json();
    if(j.error){fail(j.error);return}
    job=j.id;poll();
  }catch(e){ fail(String(e)+(mode==='up'
      ? '\n\nIf this keeps happening, copy the video to Delta once (scp) and use '
        +'"Choose on the cluster", which transfers nothing.' : '')); }

// A single 120 MB POST through an SSH tunnel and an editor port-forward is what fails;
// 8 MB pieces do not, and they give a real progress figure instead of a spinner.
async function uploadChunked(f){
  const CH=8*1024*1024;
  const r0=await fetch('/api/upload/start',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name:f.name,size:f.size})});
  const j0=await r0.json();
  if(j0.error) throw new Error(j0.error);
  let sent=0;
  for(let off=0; off<f.size; off+=CH){
    const part=f.slice(off,Math.min(off+CH,f.size));
    let ok=false, lastErr=null;
    for(let attempt=0; attempt<3 && !ok; attempt++){     // a tunnel drops the odd chunk
      try{
        const r=await fetch('/api/upload/chunk?id='+j0.id,{method:'POST',
          headers:{'Content-Type':'application/octet-stream'},body:part});
        const j=await r.json();
        if(j.error) throw new Error(j.error);
        sent=j.received; ok=true;
      }catch(e){ lastErr=e; await new Promise(s=>setTimeout(s,600*(attempt+1))); }
    }
    if(!ok) throw new Error('upload stalled at '+(sent/1048576).toFixed(0)+' MB: '+lastErr);
    const pct=100*sent/f.size;
    $('stage').textContent='Uploading '+(sent/1048576).toFixed(0)+' of '
      +(f.size/1048576).toFixed(0)+' MB';
    $('fill').style.width=pct.toFixed(1)+'%';
    $('detail').textContent=pct.toFixed(0)+'%';
  }
  $('stage').textContent='Upload complete, starting…';
  return j0.id;
}
};
function fail(m){$('prog').style.display='none';$('errc').style.display='';$('err').textContent=m;$('go').disabled=false}
async function poll(){
  const r=await fetch('/api/status?id='+job);const s=await r.json();
  $('stage').textContent=s.stage||'Working…';
  const pct=s.total?Math.min(100,100*s.frame/s.total):0;
  $('fill').style.width=pct.toFixed(1)+'%';
  $('detail').textContent=s.total?`frame ${s.frame} of ${s.total}`:'';
  if(s.state==='error'){fail(s.error);return}
  if(s.state==='done'){
    $('prog').style.display='none';$('upcard').style.display='none';$('res').style.display='';
    $('count').textContent=s.result.count;$('cand').textContent=s.result.candidates;
    $('frames').textContent=s.result.frames;$('secs').textContent=((Date.now()-t0)/1000).toFixed(0);
    $('vid').src='/api/file?id='+job+'&kind=video';
    $('dlv').href='/api/file?id='+job+'&kind=video';
    $('dlc').href='/api/file?id='+job+'&kind=csv';
    $('rule').textContent='Confirmation rule: '+s.result.rule+'  ·  source '+s.result.source;
    return;
  }
  setTimeout(poll,700);
}
</script></body></html>"""


def _safe(path: str) -> str | None:
    """Resolve `path` and return it only if it sits under an allowed root."""
    try:
        rp = os.path.realpath(os.path.expanduser(path))
    except Exception:
        return None
    for root in BROWSE_ROOTS:
        if rp == root or rp.startswith(root + os.sep):
            return rp
    return None


def list_dir(path: str) -> dict:
    rp = _safe(path or (BROWSE_ROOTS[0] if BROWSE_ROOTS else ""))
    if rp is None or not os.path.isdir(rp):
        return {"error": f"not a readable directory under the allowed roots: {path}"}
    dirs, vids = [], []
    try:
        for e in sorted(os.scandir(rp), key=lambda x: x.name.lower()):
            if e.name.startswith("."):
                continue
            if e.is_dir():
                dirs.append(e.name)
            elif e.name.lower().endswith(VIDEO_EXT):
                vids.append({"name": e.name, "size": e.stat().st_size})
    except PermissionError:
        return {"error": "permission denied"}
    parent = os.path.dirname(rp)
    return {"dir": rp, "parent": parent if _safe(parent) else None,
            "dirs": dirs, "videos": vids, "roots": BROWSE_ROOTS}


def parse_multipart(rfile, content_type: str, content_length: int, dest_dir: str):
    """Stream a multipart/form-data body to disk. -> (fields, saved_path, filename)

    Written by hand rather than with cgi.FieldStorage, which is deprecated and gone in
    Python 3.13. Streams the file part straight to disk in 1 MB chunks so a 4 GB upload
    never lands in memory, and keeps the trailing boundary-length tail back so a boundary
    split across two chunks is still found.
    """
    m = re.search(r'boundary=(?:"([^"]+)"|([^;]+))', content_type or "")
    if not m:
        raise ValueError("missing multipart boundary")
    boundary = ("--" + (m.group(1) or m.group(2)).strip()).encode()
    fields: dict[str, str] = {}
    saved_path = filename = None
    remaining = content_length

    def readline():
        nonlocal remaining
        line = rfile.readline(65536)
        remaining -= len(line)
        return line

    line = readline()
    if not line.startswith(boundary):
        raise ValueError("body does not start at the boundary")

    while remaining > 0:
        headers, name, fname = {}, None, None
        while True:
            line = readline()
            if line in (b"\r\n", b"\n", b""):
                break
            k, _, v = line.decode("utf-8", "replace").partition(":")
            headers[k.strip().lower()] = v.strip()
        disp = headers.get("content-disposition", "")
        mn = re.search(r'name="([^"]*)"', disp)
        mf = re.search(r'filename="([^"]*)"', disp)
        name = mn.group(1) if mn else None
        fname = mf.group(1) if mf else None

        if fname:                                   # the file part: stream to disk
            filename = os.path.basename(fname).replace(os.sep, "_") or "upload.mp4"
            saved_path = os.path.join(dest_dir, filename)
            tail = b""
            with open(saved_path, "wb") as out:
                while remaining > 0:
                    chunk = rfile.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    buf = tail + chunk
                    idx = buf.find(b"\r\n" + boundary)
                    if idx != -1:
                        out.write(buf[:idx])
                        rest = buf[idx + 2:]
                        done = rest.startswith(boundary + b"--")
                        # rewind logically: nothing after the boundary matters to us
                        remaining = 0 if done else remaining
                        break
                    keep = len(boundary) + 4
                    out.write(buf[:-keep] if len(buf) > keep else b"")
                    tail = buf[-keep:] if len(buf) > keep else buf
            break                                   # the video is the last part we need
        else:                                       # a small text field
            val = b""
            while remaining > 0:
                line = readline()
                if line.startswith(boundary):
                    break
                val += line
            if name:
                fields[name] = val.rstrip(b"\r\n").decode("utf-8", "replace")
    return fields, saved_path, filename


def _set(job_id: str, **kw) -> None:
    with JOBS_LOCK:
        JOBS.setdefault(job_id, {}).update(kw)


def _worker(job_id: str, video: str, out_dir: str, cfg: engine.Config) -> None:
    def progress(stage, frame, total):
        _set(job_id, stage=stage, frame=frame, total=total)
    try:
        _set(job_id, state="running", stage="Loading detector", frame=0, total=0)
        res = engine.run(video, out_dir, cfg, progress)
        _set(job_id, state="done", result=res)
    except Exception:
        _set(job_id, state="error", error=traceback.format_exc(limit=4))


class Handler(BaseHTTPRequestHandler):
    server_version = "TRACT/1.0"
    protocol_version = "HTTP/1.1"        # correct framing for a browser fetch()
    timeout = 900                        # a slow tunnel must not look like a dead client

    def log_message(self, fmt, *a):                       # keep the console readable
        if "/api/status" not in (self.path or ""):
            sys.stderr.write("  %s %s\n" % (self.command, self.path))

    # ---------------------------------------------------------------- helpers
    def _send(self, code, ctype, body: bytes, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, "application/json", json.dumps(obj).encode())

    # ---------------------------------------------------------------- routes
    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/":
            return self._send(200, "text/html; charset=utf-8", PAGE.encode())
        if p == "/api/status":
            jid = self._qs("id")
            with JOBS_LOCK:
                j = dict(JOBS.get(jid, {}))
            return self._json(j or {"state": "unknown"})
        if p == "/api/list":
            return self._json(list_dir(self._qs("dir")))
        if p == "/api/file":
            return self._file()
        return self._send(404, "text/plain", b"not found")

    def _qs(self, key):
        from urllib.parse import parse_qs, urlparse
        return parse_qs(urlparse(self.path).query).get(key, [""])[0]

    def _file(self):
        jid, kind = self._qs("id"), self._qs("kind")
        with JOBS_LOCK:
            j = dict(JOBS.get(jid, {}))
        res = j.get("result") or {}
        path = res.get("video") if kind == "video" else res.get("csv")
        if not path or not os.path.isfile(path):
            return self._send(404, "text/plain", b"no such result")
        if kind == "csv":
            ctype = "text/csv"
        else:
            ctype = "video/webm" if path.endswith(".webm") else "video/mp4"
        size = os.path.getsize(path)
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Disposition",
                         f'attachment; filename="{os.path.basename(path)}"'
                         if kind == "csv" else "inline")
        self.end_headers()
        with open(path, "rb") as fh:
            shutil.copyfileobj(fh, self.wfile, 1 << 20)


    def _upload_start(self):
        """Open a file the client will append to in chunks. -> {id}

        A whole 120 MB POST through an SSH tunnel and an editor's port forward is what
        fails in practice; several small ones do not, and they give the browser a real
        progress figure instead of a spinner.
        """
        length = int(self.headers.get("Content-Length") or 0)
        meta = json.loads(self.rfile.read(length) or b"{}")
        name = os.path.basename(str(meta.get("name") or "upload.mp4")).replace(os.sep, "_")
        if not name.lower().endswith(VIDEO_EXT):
            return self._json({"error": f"not a video file: {name}"}, 400)
        size = int(meta.get("size") or 0)
        if size > MAX_BYTES:
            return self._json({"error": f"file exceeds {MAX_BYTES // 1024**3} GB"}, 413)
        uid = uuid.uuid4().hex[:12]
        d = os.path.join(WORK, "up_" + uid)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, name)
        open(path, "wb").close()
        with JOBS_LOCK:
            UPLOADS[uid] = {"path": path, "name": name, "size": size, "received": 0}
        return self._json({"id": uid})

    def _upload_chunk(self):
        uid = self._qs("id")
        with JOBS_LOCK:
            up = UPLOADS.get(uid)
        if not up:
            return self._json({"error": "unknown upload id"}, 400)
        length = int(self.headers.get("Content-Length") or 0)
        if length > CHUNK_LIMIT:
            return self._json({"error": "chunk too large"}, 413)
        got = 0
        with open(up["path"], "ab") as fh:
            while got < length:
                b = self.rfile.read(min(1 << 20, length - got))
                if not b:
                    break
                fh.write(b)
                got += len(b)
        with JOBS_LOCK:
            up["received"] += got
            received = up["received"]
        return self._json({"received": received})

    def do_POST(self):
        route = self.path.split("?")[0]
        if route == "/api/upload/start":
            return self._upload_start()
        if route == "/api/upload/chunk":
            return self._upload_chunk()
        if route != "/api/run":
            return self._send(404, "text/plain", b"not found")
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BYTES:
                return self._json({"error": f"upload exceeds {MAX_BYTES // 1024**3} GB"}, 413)
            jid = uuid.uuid4().hex[:12]
            job_dir = os.path.join(WORK, jid)
            os.makedirs(job_dir, exist_ok=True)
            ctype = self.headers.get("Content-Type", "")
            if ctype.startswith("application/json"):
                fields = json.loads(self.rfile.read(length) or b"{}")
                src = None
            else:
                fields, src, fname = parse_multipart(self.rfile, ctype, length, job_dir)
            uid = str(fields.get("upload_id") or "").strip()
            if uid:
                with JOBS_LOCK:
                    up = UPLOADS.pop(uid, None)
                if not up or not os.path.isfile(up["path"]):
                    return self._json({"error": "upload not found; start it again"}, 400)
                if up["size"] and up["received"] != up["size"]:
                    return self._json({"error": f"upload incomplete: {up['received']} of "
                                                f"{up['size']} bytes arrived"}, 400)
                src = up["path"]
            if not src:
                # No upload: the caller named a file that already lives on this filesystem,
                # which is the normal case when the browser and the data share a cluster.
                # It skips copying the video entirely.
                given = (fields.get("path") or "").strip()
                if not given:
                    return self._json({"error": "no video in the request"}, 400)
                src = _safe(given)
                if src is None or not os.path.isfile(src):
                    return self._json(
                        {"error": f"not a readable video under the allowed roots:\n{given}"}, 400)
                if not src.lower().endswith(VIDEO_EXT):
                    return self._json({"error": f"not a video file: {src}"}, 400)
            elif not os.path.getsize(src):
                return self._json({"error": "uploaded file was empty"}, 400)

            cfg = engine.Config(**vars(CFG))
            def num(name, cast, lo, hi, dflt):
                try:
                    v = cast(fields.get(name, dflt))
                except (TypeError, ValueError):
                    return dflt
                return max(lo, min(hi, v))
            cfg.conf = num("conf", float, 0.01, 0.9, 0.10)
            cfg.min_frames = num("min_frames", int, 1, 500, 20)
            cfg.min_topk_conf = num("min_topk", float, 0.0, 1.0, 0.65)
            cfg.draw_unconfirmed = fields.get("draw_candidates", "1") == "1"

            _set(jid, state="queued", stage="Queued", frame=0, total=0)
            threading.Thread(target=_worker, args=(jid, src, job_dir, cfg),
                             daemon=True).start()
            return self._json({"id": jid})
        except Exception:
            return self._json({"error": traceback.format_exc(limit=4)}, 500)


def main() -> None:
    global WORK, BROWSE_ROOTS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--weights", default=engine.DEFAULT_WEIGHTS)
    ap.add_argument("--device", default="0", help="'0' for the first GPU, 'cpu' to force CPU")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--work", default=WORK)
    ap.add_argument("--roots", default=os.path.join(_PROJ, "data") + ",/work/hdd/bgte/tislam6,"
                                      "/work/nvme/bgte/tislam6",
                    help="comma-separated directories the browser may read videos from")
    args = ap.parse_args()

    WORK = args.work
    os.makedirs(WORK, exist_ok=True)
    CFG.weights, CFG.device, CFG.imgsz = args.weights, args.device, args.imgsz
    global BROWSE_ROOTS
    BROWSE_ROOTS = [os.path.realpath(os.path.expanduser(r))
                    for r in args.roots.split(",") if r.strip()]
    if not os.path.isfile(CFG.weights):
        raise SystemExit(f"weights not found: {CFG.weights}")
    if args.device != "cpu":
        try:
            import torch
            if not torch.cuda.is_available():
                print("[warn] no CUDA device visible. On a login node use --device cpu, "
                      "or request a GPU with srun.", file=sys.stderr)
        except Exception:
            pass

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    host = os.uname().nodename
    print(f"TRACT interface on http://{host}:{args.port}  (device {args.device})")
    print(f"  weights : {CFG.weights}")
    print(f"  workdir : {WORK}")
    print(f"  roots   : {', '.join(BROWSE_ROOTS)}")
    print(f"  tunnel  : ssh -L {args.port}:{host}:{args.port} <user>@login.delta.ncsa.illinois.edu")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
