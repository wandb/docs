#!/usr/bin/env bash
# Export marimo cookbooks under weave/cookbooks/source/ to Markdown (.mdx) for
# inclusion in weave/cookbooks/*.mdx via import.
#
# Requires: marimo CLI (pip install marimo)
#
# Usage:
#   ./scripts/export-marimo-cookbook-md.sh
#   ./scripts/export-marimo-cookbook-md.sh path/to/notebook.py   # single file
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
SOURCE_DIR="${ROOT}/weave/cookbooks/source"
OUT_DIR="${ROOT}/weave/cookbooks/generated"

if ! command -v marimo >/dev/null 2>&1; then
  echo "error: marimo not found on PATH. Install with: pip install marimo" >&2
  exit 1
fi

process_export() {
  local out_path="$1"
  python3 - "${out_path}" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")

# Strip YAML frontmatter (marimo adds title / marimo-version).
lines = text.splitlines(keepends=True)
if lines and lines[0].strip() == "---":
    i = 1
    while i < len(lines) and lines[i].strip() != "---":
        i += 1
    if i < len(lines):
        text = "".join(lines[i + 1 :])

# Remove Docusaurus meta block if present.
text = re.sub(
    r"<!--\s*docusaurus_head_meta::start.*?docusaurus_head_meta::end\s*-->",
    "",
    text,
    flags=re.DOTALL,
)

# MDX treats `{...}` as expressions in some parsers; marimo uses `{.marimo}` on fences.
text = text.replace(" {.marimo}", "")

# Marimo cell markers and HTML comments: `<!---->` / `<!-- ... -->` break MDX parsing.
text = re.sub(r"<!---->\s*", "", text)
text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

# Self-close img tags (marimo exports </img> which MDX rejects).
text = re.sub(r"<img([^>]+)>\s*</img>", r"<img\1 />", text, flags=re.IGNORECASE)

# Remove boilerplate cells that are only `import marimo as mo` (noise in docs).
text = re.sub(
    r"```python\s*\nimport marimo as mo\s*\n```\s*\n",
    "",
    text,
)
text = text.lstrip("\n")

path.write_text(text, encoding="utf-8")
PY
}

export_one() {
  local py_path="$1"
  local base
  base="$(basename "${py_path}" .py)"
  local out_path="${OUT_DIR}/${base}.mdx"
  mkdir -p "${OUT_DIR}"
  echo "marimo export md: ${py_path} -> ${out_path}"
  marimo export md "${py_path}" -o "${out_path}" -f
  process_export "${out_path}"
}

if [[ $# -gt 0 ]]; then
  for f in "$@"; do
    export_one "${f}"
  done
  exit 0
fi

mkdir -p "${OUT_DIR}"

# Only notebooks that have a matching weave/cookbooks/<name>.mdx page.
shopt -s nullglob
for mdx in "${ROOT}/weave/cookbooks/"*.mdx; do
  base="$(basename "${mdx}" .mdx)"
  py="${SOURCE_DIR}/${base}.py"
  if [[ -f "${py}" ]]; then
    export_one "${py}"
  else
    echo "skip (no source): ${base}.mdx" >&2
  fi
done

cat > "${OUT_DIR}/README.md" <<'EOF'
# Generated marimo Markdown

These `.mdx` files are produced by `marimo export md` from notebooks in
`../source/`. Regenerate after editing a notebook:

```bash
./scripts/export-marimo-cookbook-md.sh
```

Do not edit generated files by hand.
EOF

echo "done. Output: ${OUT_DIR}"
