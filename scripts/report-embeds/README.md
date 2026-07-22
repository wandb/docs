# Report embeds

`check_embeds.py` keeps live W&B report embeds honest. It finds every
[`<WandbReport>`](../../snippets/WandbReport.jsx) by scanning the English `.mdx`
sources (no registry), then checks each one: it sits on a page, has a
recognizable report URL, and still renders anonymously over the network.

## Embed a report

Use a **regular W&B report, not a Fully Connected article** (FC articles keep
their full blog chrome in a frame and look broken), viewable anonymously — in a
public project or shared via a view-only link (`Share` → "anyone with the link
can view"). The URL, including any `?accessToken=`, ships in public source and
git history, so treat the report as public forever. Add the component (it
renders a "View Report" button, so no separate link is needed):

```mdx
import { WandbReport } from '/snippets/WandbReport.jsx';

<WandbReport
  src="https://wandb.ai/ENTITY/PROJECT/reports/Slug--VmlldzoXXXXXXX"
  title="Accessible description of the report"
  height={700}
/>
```

## Run the checks

```bash
python3 scripts/report-embeds/check_embeds.py            # scan + liveness
python3 -m unittest discover -s scripts/report-embeds -p 'check_embeds.py'  # unit tests
```

CI ([`report-embeds.yml`](../../.github/workflows/report-embeds.yml)) runs this on
PRs that touch embeds and weekly (filing an issue if a report has gone dead).

## Component coupling

`check_embeds.py` keys on the literal tag `WandbReport` and the `src` prop
(`COMPONENT_RE` / `SRC_ATTR_RE`). If `snippets/WandbReport.jsx` renames either,
update those constants.
