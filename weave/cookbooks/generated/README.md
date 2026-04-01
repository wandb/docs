# Generated marimo Markdown

These `.mdx` files are produced by `marimo export md` from notebooks in
`../source/`. Regenerate after editing a notebook:

```bash
./scripts/export-marimo-cookbook-md.sh
```

Do not edit generated files by hand.

The export strips YAML frontmatter, marimo fence metadata (`{.marimo}`), marimo and HTML comments, normalizes `img` tags for MDX, and removes boilerplate fenced cells that only contain `import marimo as mo`.

Some notebooks still contain relative links and images (for example `../../media/...`) that point at paths from the old docs layout. `mint broken-links` may list those until the notebook source is updated or paths are rewritten.
