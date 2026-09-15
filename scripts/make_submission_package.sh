#!/usr/bin/env bash
# Build the Neural Computing and Applications submission package from the Springer Overleaf tree.
#
#   bash scripts/make_submission_package.sh <overleaf_tree> <out_dir>
#
# Springer's Editorial Manager wants (a) ONE .tex file with no \input{}, (b) every source file
# it needs (class, bst, bib/bbl, figures), (c) the blinded PDF, (d) the title page as a
# separate file. This script flattens main.tex, copies only the figures actually used,
# compiles the result to prove it builds stand-alone, checks the blinded PDF for author
# metadata, and zips the lot. Requires pdflatex/bibtex on PATH (TinyTeX: ~/.TinyTeX/bin).
set -euo pipefail
SRC=${1:?overleaf tree}; OUT=${2:?output dir}
STAMP=$(date +%Y-%m-%d)
PKG=$OUT/NCA_submission_$STAMP
rm -rf "$PKG" && mkdir -p "$PKG/figures"

python3 - "$SRC" "$PKG" <<'EOF'
import re, sys, os, shutil
src, pkg = sys.argv[1:3]
def flatten(path, depth=0):
    assert depth < 5, "input nesting too deep"
    out = []
    for line in open(os.path.join(src, path)):
        m = re.match(r'\s*\\input\{([^}]+)\}\s*$', line)
        if m:
            sub = m.group(1) if m.group(1).endswith('.tex') else m.group(1) + '.tex'
            out.append(f'%% ---- begin {sub} ----\n'); out += flatten(sub, depth + 1); out.append(f'%% ---- end {sub} ----\n')
        else:
            out.append(line)
    return out
main = ''.join(flatten('main.tex'))
assert not re.search(r'^[^%\n]*\\input\{', main, re.M), "an \\input survived flattening"  # comments may mention \input
assert '\\blindtrue' in main, "main.tex is not set to \\blindtrue: the review copy must be anonymised"
open(os.path.join(pkg, 'main.tex'), 'w').write(main)
figs = set(re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', main))
for f in sorted(figs):
    cand = [f] + [f + ext for ext in ('.pdf', '.png', '.jpg', '.jpeg', '.eps')]
    hit = next((c for c in cand if os.path.exists(os.path.join(src, c))), None)
    assert hit, f"figure not found: {f}"
    os.makedirs(os.path.dirname(os.path.join(pkg, hit)), exist_ok=True)
    shutil.copy(os.path.join(src, hit), os.path.join(pkg, hit))
for f in ('sn-jnl.cls', 'sn-basic.bst', 'main.bib', 'title_page.tex'):
    shutil.copy(os.path.join(src, f), pkg)
print(f"flattened main.tex ({len(main.splitlines())} lines), {len(figs)} figures copied")
EOF

cd "$PKG"
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
bibtex main >/dev/null
pdflatex -interaction=nonstopmode main.tex >/dev/null
pdflatex -interaction=nonstopmode main.tex > build.log
pdflatex -interaction=nonstopmode -halt-on-error title_page.tex >/dev/null
echo "stand-alone build: errors=$(grep -c '^!' build.log || true) undefined=$(grep -c -i 'undefined' build.log || true) $(grep -o 'Output written on main.pdf ([0-9]* pages' build.log)"
rm -f *.aux *.log *.out *.blg *.toc *.spl title_page.aux
# keep main.bbl: Editorial Manager can then build without running BibTeX

python3 - <<'EOF'
import re
pdf = open('main.pdf', 'rb').read()
for key in (b'/Author', b'/Title'):
    m = re.search(key + rb'\s*\((.*?)\)', pdf)
    print(key.decode(), ':', m.group(1)[:80] if m else 'not set')
for name in (b'Islam', b'Sarker', b'Morelock', b'Bastille', b'Ahmed', b'siu.edu'):
    if name in pdf:
        print('WARNING: blinded PDF contains', name.decode())
EOF

cd "$OUT"
rm -f "NCA_submission_$STAMP.zip"
zip -q -r "NCA_submission_$STAMP.zip" "NCA_submission_$STAMP" -x '*.DS_Store'
echo "package: $OUT/NCA_submission_$STAMP.zip ($(du -h "NCA_submission_$STAMP.zip" | cut -f1))"
ls "$PKG"
