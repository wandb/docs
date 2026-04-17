#!/usr/bin/env bash
# Generate QEL reference MDX from wandb/weave-internal weave-js (see README.md).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# query-panel -> reference-generation -> scripts -> repo root
DOCS_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
WEAVE_JS="${WEAVE_JS_ROOT:-}"
CLEANUP_CLONE=""

if [[ -z "${WEAVE_JS}" ]]; then
  TMP="$(mktemp -d)"
  CLEANUP_CLONE="1"
  cleanup() {
    if [[ -n "${CLEANUP_CLONE}" ]]; then
      rm -rf "${TMP}"
    fi
  }
  trap cleanup EXIT

  if [[ -n "${QUERY_PANELS_READ_TOKEN:-}" ]]; then
    CLONE_URL="https://x-access-token:${QUERY_PANELS_READ_TOKEN}@github.com/wandb/weave-internal.git"
  else
    CLONE_URL="https://github.com/wandb/weave-internal.git"
  fi

  echo "Cloning wandb/weave-internal (shallow)..."
  export GIT_TERMINAL_PROMPT=0
  git \
    -c http.https://github.com/.extraheader= \
    -c credential.helper= \
    clone --depth 1 "${CLONE_URL}" "${TMP}/weave-internal" --quiet

  WEAVE_JS="${TMP}/weave-internal/weave-js"
fi

if [[ ! -f "${WEAVE_JS}/package.json" ]]; then
  echo "error: weave-js not found at ${WEAVE_JS} (set WEAVE_JS_ROOT to weave-js directory)" >&2
  exit 1
fi

README_SRC="${WEAVE_JS}/src/core/docs/README.md"
if [[ ! -f "${README_SRC}" ]]; then
  echo "error: missing ${README_SRC}" >&2
  exit 1
fi

mkdir -p "${WEAVE_JS}/docs"
cp "${README_SRC}" "${WEAVE_JS}/docs/README.md"

echo "Installing yarn dependencies (ignore scripts) under ${WEAVE_JS}..."
(
  cd "${WEAVE_JS}"
  yarn install --ignore-scripts
)

echo "Running upstream generateDocs (vite-node)..."
(
  cd "${WEAVE_JS}"
  export NODE_OPTIONS="${NODE_OPTIONS:---max-old-space-size=8192}"
  # Pin vite-node major to match weave-js Vite 3.x toolchain.
  npx --yes vite-node@3.1.3 src/core/generateDocs.ts
)

DOCS_GEN="${WEAVE_JS}/docs_gen"
if [[ ! -d "${DOCS_GEN}" ]]; then
  echo "error: docs_gen not produced at ${DOCS_GEN}" >&2
  exit 1
fi

echo "Converting Markdown to Mintlify MDX..."
python3 "${SCRIPT_DIR}/convert_query_panel_md.py" \
  --docs-gen "${DOCS_GEN}" \
  --out-dir "${DOCS_ROOT}/models/ref/query-panel" \
  --landing "${DOCS_ROOT}/models/ref/query-panel.mdx" \
  --docs-json "${DOCS_ROOT}/docs.json"

echo "Done. Review changes under models/ref/query-panel/ and models/ref/query-panel.mdx"
