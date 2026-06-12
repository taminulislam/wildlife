"""
Parse FLIR video paths into structured metadata.

The archive is laid out as::

    data/raw/<SITE>_Videos_LS/visit<N>/[<SITE>_<date>/]<Transect>_<SITE>_<Date>_<Side>.mp4
    e.g. data/raw/MAS_videos_LS/Visit1/EMarseilles_MAS_12.22.25_LS.mp4

So the real fields are:
  * **site**   = the 3-4 letter code (MAS, SHB, SHW, TON) — taken from the
                 ``<SITE>_Videos_LS`` folder, or the 2nd filename token. This is the
                 unit we stratify and split on. NOTE: the *first* filename token is the
                 **transect/road name** (EMarseilles, GolfDr, ...), NOT the site.
  * **transect** = the road/route name (first filename token).
  * **visit**  = visit number if the path has a ``visitN`` folder. Per project decision
                 we do NOT split or group by visit (visits are combined into the site);
                 it is kept only to disambiguate files and for provenance.
  * **side**   = LS/RS (left/right window).

``key`` is a collision-proof, readable id for a single video — needed because the same
transect filename can recur across visits (e.g. MAS EMarseilles in Visit1 and Visit2).
The segment before the first ``__`` is always the site, so it can be recovered from a
frame filename later.
"""
from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass

_DATE_RE = re.compile(
    r"(?P<d>\d{1,2}[._-]\d{1,2}[._-]\d{2,4}|\d{4}[._-]\d{1,2}[._-]\d{1,2})"
)
_SIDE_RE = re.compile(r"(?:^|[_-])(?P<side>LS|RS)(?:$|[_-])", re.IGNORECASE)
_SITE_FOLDER_RE = re.compile(r"^(?P<site>[A-Za-z]{2,5})_videos?_(?P<side>LS|RS)$", re.I)
_VISIT_RE = re.compile(r"visit\s*0*(?P<n>\d+)", re.I)
_DATE_FOLDER_RE = re.compile(r"^[A-Za-z]{2,5}_\d{1,2}[._-]\d{1,2}[._-]\d{2,4}$")

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".ts")


def _sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", s)


@dataclass
class VideoMeta:
    path: str          # original path as given
    filename: str
    stem: str
    site: str
    transect: str
    date: str
    side: str          # "LS", "RS", or ""
    visit: str         # "1", "2", or ""

    @property
    def key(self) -> str:
        """Collision-proof, readable id. Segment before '__' is the site."""
        parts = [_sanitize(self.transect)]
        if self.visit:
            parts.append(f"v{self.visit}")
        if self.side:
            parts.append(self.side)
        return f"{self.site}__" + "_".join(parts)

    @property
    def site_key(self) -> str:
        return self.site

    def as_dict(self) -> dict:
        d = asdict(self)
        d["key"] = self.key
        return d


def parse_path(path: str) -> VideoMeta:
    """Parse a full/relative path (preferred) or a bare filename into VideoMeta."""
    norm = path.replace("\\", "/")
    parts = norm.split("/")
    base = parts[-1]
    stem = os.path.splitext(base)[0]
    tokens = stem.split("_")

    # side: prefer filename, fall back to the site folder's suffix.
    side = ""
    m = _SIDE_RE.search(stem)
    if m:
        side = m.group("side").upper()

    # site + side from an ancestor "<SITE>_Videos_LS" folder, if present.
    site = ""
    for p in parts[:-1]:
        fm = _SITE_FOLDER_RE.match(p)
        if fm:
            site = fm.group("site").upper()
            if not side:
                side = fm.group("side").upper()
            break

    # date
    date = ""
    md = _DATE_RE.search(stem)
    if md:
        date = md.group("d")

    # visit number from any "visitN" ancestor folder.
    visit = ""
    for p in parts[:-1]:
        vm = _VISIT_RE.search(p)
        if vm:
            visit = vm.group("n")
            break

    # transect = first filename token (the road/route name).
    transect = tokens[0] if tokens else stem

    # Fallback site: 2nd filename token if it's a short alpha code (not side/date).
    if not site:
        for tok in tokens[1:]:
            if tok.upper() in ("LS", "RS") or _DATE_RE.search(tok):
                continue
            if tok.isalpha() and 2 <= len(tok) <= 5:
                site = tok.upper()
                break
    if not site:
        site = _sanitize(transect) or "UNKNOWN"

    return VideoMeta(path=path, filename=base, stem=stem, site=site,
                     transect=transect, date=date, side=side, visit=visit)


def site_from_key(key: str) -> str:
    return key.split("__", 1)[0] if "__" in key else key


def iter_videos(raw_dir: str):
    """Yield (path, VideoMeta) for every video under raw_dir (recursive), sorted."""
    found: list[str] = []
    for root, _dirs, files in os.walk(raw_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in VIDEO_EXTS:
                found.append(os.path.join(root, f))
    for p in sorted(found):
        yield p, parse_path(p)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        targets = sys.argv[1:]
        for t in targets:
            m = parse_path(t)
            print(f"{t}\n   site={m.site!r} transect={m.transect!r} visit={m.visit!r} "
                  f"side={m.side!r} date={m.date!r} key={m.key!r}")
    else:
        # Summarize the real archive grouped by site.
        from collections import defaultdict
        by_site: dict[str, list[str]] = defaultdict(list)
        keys = set()
        for p, m in iter_videos("data/raw"):
            by_site[m.site].append(m.key)
            if m.key in keys:
                print(f"!! KEY COLLISION: {m.key} ({p})")
            keys.add(m.key)
        print(f"{sum(len(v) for v in by_site.values())} videos across "
              f"{len(by_site)} sites:")
        for site, ks in sorted(by_site.items()):
            print(f"  {site}: {len(ks)} videos")
