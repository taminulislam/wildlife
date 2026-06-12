"""
Parse FLIR video filenames into structured metadata.

Filenames follow roughly: ``<Site>_<Observer>_<Date>_<Side>.mp4``
e.g. ``EagleCreekDrEtc_SHB_12.11.2025_LS.mp4``
  -> site="EagleCreekDrEtc", observer="SHB", date="12.11.2025", side="LS"

Real-world filenames are messy, so the parser is defensive: it pulls out the side
(LS/RS) and a date wherever they appear, treats the leading token as the site, and
always keeps the raw stem so nothing is lost. ``visit_key`` groups the (site, date)
pairs that should be treated as one survey visit — we expect 2 visits per site.

This metadata drives stratified sampling and, crucially, **splitting train/val/test by
site** so the model is evaluated on locations it never trained on.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

# Matches dates like 12.11.2025, 12-11-2025, 12_11_2025, 2025.12.11
_DATE_RE = re.compile(
    r"(?P<d>\d{1,2}[._-]\d{1,2}[._-]\d{2,4}|\d{4}[._-]\d{1,2}[._-]\d{1,2})"
)
_SIDE_RE = re.compile(r"(?:^|[_-])(?P<side>LS|RS)(?:$|[_-])", re.IGNORECASE)


@dataclass
class VideoMeta:
    filename: str
    stem: str
    site: str
    observer: str
    date: str
    side: str  # "LS", "RS", or "" if not found

    @property
    def visit_key(self) -> str:
        """Stable key for a single survey visit (site + date)."""
        return f"{self.site}__{self.date}" if self.date else self.site

    @property
    def site_key(self) -> str:
        return self.site

    def as_dict(self) -> dict:
        d = asdict(self)
        d["visit_key"] = self.visit_key
        return d


def parse_filename(filename: str) -> VideoMeta:
    """Parse one filename (with or without directory / extension) into VideoMeta."""
    import os

    base = os.path.basename(filename)
    stem = os.path.splitext(base)[0]

    side = ""
    m = _SIDE_RE.search(stem)
    if m:
        side = m.group("side").upper()

    date = ""
    md = _DATE_RE.search(stem)
    if md:
        date = md.group("d")

    # Site = leading token before the first underscore; falls back to whole stem.
    tokens = stem.split("_")
    site = tokens[0] if tokens else stem

    # Observer = a short alpha token that is not the side and not part of the date.
    observer = ""
    for tok in tokens[1:]:
        if _SIDE_RE.fullmatch(f"_{tok}_") or _DATE_RE.search(tok):
            continue
        if tok.isalpha() and 2 <= len(tok) <= 5:
            observer = tok
            break

    return VideoMeta(
        filename=base, stem=stem, site=site, observer=observer, date=date, side=side
    )


if __name__ == "__main__":
    import sys

    samples = sys.argv[1:] or [
        "EagleCreekDrEtc_SHB_12.11.2025_LS.mp4",
        "TON_JD_01.03.2026_RS.mp4",
        "weird-name.mp4",
    ]
    for s in samples:
        m = parse_filename(s)
        print(f"{s}")
        print(f"   site={m.site!r} observer={m.observer!r} date={m.date!r} "
              f"side={m.side!r} visit_key={m.visit_key!r}")
