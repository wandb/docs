---
title: "Tutorial Name # 1"
format: docusaurus-md
---

export const quartoRawHtml =
[`
wandb version 0.13.10 is available!  To upgrade, please run:
 $ pip install wandb --upgrade
`,`
Tracking run with wandb version 0.13.9
`,`
Run data is saved locally in <code>/Users/noahluna/Documents/GitHub/docodile/docs/tutorials/wandb/run-20230225_114107-oim5ml1v</code>
`,`
Syncing run <strong><a href="https://wandb.ai/noahluna/my-awesome-project/runs/oim5ml1v" target="_blank">rich-oath-4</a></strong> to <a href="https://wandb.ai/noahluna/my-awesome-project" target="_blank">Weights & Biases</a> (<a href="https://wandb.me/run" target="_blank">docs</a>)<br/>
`,`
 View project at <a href="https://wandb.ai/noahluna/my-awesome-project" target="_blank">https://wandb.ai/noahluna/my-awesome-project</a>
`,`
 View run at <a href="https://wandb.ai/noahluna/my-awesome-project/runs/oim5ml1v" target="_blank">https://wandb.ai/noahluna/my-awesome-project/runs/oim5ml1v</a>
`,`
Waiting for W&B process to finish... <strong style="color:green">(success).</strong>
`,`
 View run <strong style="color:#cdcd00">rich-oath-4</strong> at: <a href="https://wandb.ai/noahluna/my-awesome-project/runs/oim5ml1v" target="_blank">https://wandb.ai/noahluna/my-awesome-project/runs/oim5ml1v</a><br/>Synced 5 W&B file(s), 0 media file(s), 2 artifact file(s) and 1 other file(s)
`,`
Find logs at: <code>./wandb/run-20230225_114107-oim5ml1v/logs</code>
`];

## This was made with a notebook {#this-was-made-with-a-notebook}

``` python
print("Hello, world!")
```

``` text
Hello, world!
```

Create your first W&B Run:

``` python
import wandb

run = wandb.init(project='my-awesome-project')

run.finish()
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[0] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[1] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[2] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[3] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[4] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[5] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[6] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[7] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[8] }} />

Then we explain stuff.

## This is another section. {#this-is-another-section.}
