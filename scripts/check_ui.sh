#!/usr/bin/env bash
# Syntax-check the JavaScript embedded in the TRACT page.
#
# Worth having: the page is a Python string, so a Python escape can silently corrupt a JS
# string literal. That happened once -- Python ate the \n in an error message and put a real
# newline inside a JS string, which broke the whole script and made every button dead with
# no server-side symptom at all.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/work/nvme/bgte/tislam6/envs/wildlife/bin/python
JS=$(mktemp /tmp/tract_ui.XXXXXX.js)
"$PY" - "$ROOT" "$JS" <<'PYEOF'
import re, sys
sys.path.insert(0, sys.argv[1] + "/src/app")
import server
m = re.search(r"<script>(.*?)</script>", server.PAGE, re.S)
if not m:
    sys.exit("no <script> block found in PAGE")
open(sys.argv[2], "w").write(m.group(1))
print(f"extracted {len(m.group(1))} chars of JS")
PYEOF
if command -v node >/dev/null; then
  node --check "$JS" && echo "JS syntax OK"
else
  echo "node not on PATH; skipped the syntax check"
fi
rm -f "$JS"
