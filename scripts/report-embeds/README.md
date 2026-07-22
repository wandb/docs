# Report embeds

Tooling that keeps live W&B report embeds honest. A report embedded with the
[`<WandbReport>`](../../snippets/WandbReport.jsx) component is database content:
it can drift or vanish without a docs PR. `check_embeds.py` finds every embed by
scanning the `.mdx` sources (no registry to maintain) and checks it.

## Files

- `check_embeds.py` — scans English `.mdx` for `<WandbReport>` embeds and checks
  them (sibling prose link + per-page cap + URL liveness).
- `tests/test_check_embeds.py` — unit tests (no network).

CI wiring lives in [`.github/workflows/report-embeds.yml`](../../.github/workflows/report-embeds.yml).

## Add an embedded report

1. Confirm the report is a **regular W&B report, not a Fully Connected article**
   (FC articles keep their full blog chrome in a frame and look broken), and that
   it is viewable by anonymous visitors — a report in a public project, or one
   shared via a view-only ("magic") link (`Share` → "anyone with the link can
   view"). The URL, including any `?accessToken=`, ships in public source and git
   history, so treat the report as public forever. Never embed sensitive data.
2. On the page, add both the component and a plain Markdown link to the same
   report in the surrounding prose:

   ```mdx
   import { WandbReport } from '/snippets/WandbReport.jsx';

   The following [sweep report](https://wandb.ai/ENTITY/PROJECT/reports/Slug--VmlldzoXXXXXXX)
   shows ... .

   <WandbReport
     src="https://wandb.ai/ENTITY/PROJECT/reports/Slug--VmlldzoXXXXXXX"
     title="Accessible description of the report"
     height={700}
   />
   ```

   The prose link is required: agents, the llms.txt export, and the translation
   pipeline read MDX source, where the iframe is opaque.

3. Validate locally:

   ```bash
   python3 scripts/report-embeds/check_embeds.py --mode static
   ```

## Run the checks

```bash
python3 -m unittest discover -s scripts/report-embeds/tests -p 'test_*.py' -v
python3 scripts/report-embeds/check_embeds.py --mode static     # prose link + cap, no network
python3 scripts/report-embeds/check_embeds.py --mode liveness   # anonymous HTTP per URL
python3 scripts/report-embeds/check_embeds.py --mode all         # both
```

Static errors (missing prose link, embed in a shared snippet) fail CI on PRs.
Liveness runs on a weekly schedule and files an issue on failure — it does not
block PRs, because `wandb.ai` rate-limits crawlers and network results are flaky
(this is also why `wandb.ai` is excluded from the site-wide lychee check).

## Component coupling

`check_embeds.py` keys on the literal tag `WandbReport` and the `src` prop
(`COMPONENT_RE` / `SRC_ATTR_RE` at the top of the file). If
`snippets/WandbReport.jsx` renames either, update those constants.
