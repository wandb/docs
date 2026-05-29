# Readability delta check

An informational, non-blocking CI check that reports how a pull request affects
the readability of the docs it touches (DOCS-2626). It posts a single PR comment
with a word-weighted Flesch-Kincaid grade delta across the changed pages, plus an
optional AI-agent-comprehension rating.

We report the *delta*, not an absolute grade. Technical docs have an inherently
high reading grade level, so "this page went from 12.4 to 11.1" is meaningful
where "12.4" on its own is not. A lower Flesch-Kincaid grade and a higher Flesch
reading ease both mean easier to read.

## How it works

The scoring logic lives in the `coreweave/docs-skills` submodule (`.claude/`),
shared with `coreweave/documentation`:

- `.claude/scripts/_readability.py` extracts narrative prose from MDX (stripping
  frontmatter, code, JSX/Mintlify components, tables, and links) and scores it
  with `textstat` (Flesch reading ease, Flesch-Kincaid grade, Gunning fog, SMOG).
- `.claude/scripts/_docs_eval_lib.py` provides the AI-agent-comprehension LLM
  judge (W&B Inference, authenticated with `WANDB_API_KEY`).
- `.claude/scripts/readability_baselines.json` holds per-doc-type baseline grade
  levels (regenerate with `.claude/scripts/readability_baselines.py`).

This directory holds the wandb/docs plumbing:

- `pr_report.py` diffs the PR base and head, scores each changed English `.mdx`
  file, aggregates a word-weighted delta, and builds the Markdown report.
- `tests/` covers the parsing and Markdown-building logic (no network needed).

The workflow is `.github/workflows/readability-delta.yml`.

## Behavior

- **Non-blocking**: the check never fails a PR. It only posts a comment.
- **Forks**: skipped (no access to the submodule or W&B Inference); the workflow
  posts an Actions notice and exits successfully.
- **Deterministic backbone**: the `textstat` delta always runs. The LLM judge is
  supplementary and degrades gracefully when the W&B Inference key is absent. In
  CI the key comes from the `WANDB_DOCS_INFERENCE_API_KEY` repo secret, passed to
  the scorer as `WANDB_API_KEY`.
- **Localized content** under `ja/`, `ko/`, and `fr/` is skipped.

## Run locally

```bash
pip install -r scripts/readability/requirements.txt
git submodule update --init .claude
python3 scripts/readability/pr_report.py --base main --head HEAD
```

Add `--judge` to include the comprehension judge (requires `WANDB_API_KEY`).
