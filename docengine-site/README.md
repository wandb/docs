# docengine-site/

DocEngine **source** for this repo: the slim site descriptor (`docengine.yaml` — `id` +
`output_subpath`) plus `schemas/`, `data/`, `templates/`, `generators/`, `importers/`, and
`overlays/`. The engine itself is the `docengine/` submodule.

The `docengine-site/` name is a fixed convention — its presence (specifically
`docengine-site/docengine.yaml`) marks this directory's parent as the DocEngine **host root**.

This is a minimal "Hello World" install:

- `schemas/greeting.py` — the `greeting` table (one row per `data/greeting/*.yaml`).
- `data/greeting/world.yaml` — a single published greeting row.
- `templates/hello_world_snippet.mdx.j2` — the snippet template.
- `generators/hello_world_snippet.py` — renders the snippet from `greeting` rows.
- `importers/` and `overlays/` — empty stubs (`.gitkeep`) for now.

## Building

This repo sets `output_subpath: .`, so generated output lands in the repo root's
Mintlify tree. Locally, run `docengine build` from the repo root after `direnv allow`
(or `uv sync --project ./docengine --extra dev`); it regenerates
`snippets/docengine/hello-world.mdx`. A push touching `docengine-site/` (or the
`docengine` submodule pin) runs `.github/workflows/docengine-build.yml`, which builds
and commits the regenerated snippet back to the same branch.
