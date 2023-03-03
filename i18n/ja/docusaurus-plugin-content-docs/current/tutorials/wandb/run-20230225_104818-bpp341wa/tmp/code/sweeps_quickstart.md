export const quartoRawHtml =
[`
wandb version 0.13.10 is available!  To upgrade, please run:
 $ pip install wandb --upgrade
`,`
Tracking run with wandb version 0.13.9
`,`
Run data is saved locally in <code>/Users/noahluna/Documents/GitHub/docodile/docs/tutorials/wandb/run-20230225_104818-bpp341wa</code>
`,`
Syncing run <strong><a href="https://wandb.ai/noahluna/my-awesome-project/runs/bpp341wa" target="_blank">resilient-shadow-1</a></strong> to <a href="https://wandb.ai/noahluna/my-awesome-project" target="_blank">Weights & Biases</a> (<a href="https://wandb.me/run" target="_blank">docs</a>)<br/>
`,`
 View project at <a href="https://wandb.ai/noahluna/my-awesome-project" target="_blank">https://wandb.ai/noahluna/my-awesome-project</a>
`,`
 View run at <a href="https://wandb.ai/noahluna/my-awesome-project/runs/bpp341wa" target="_blank">https://wandb.ai/noahluna/my-awesome-project/runs/bpp341wa</a>
`];

``` python
!pip install wandb
```

``` python
# Import the W&B Python Library and log into W&B
import wandb
wandb.login()

run = wandb.init(project='my-awesome-project')

```

``` text
wandb: Currently logged in as: noahluna. Use `wandb login --relogin` to force relogin
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[0] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[1] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[2] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[3] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[4] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[5] }} />

``` python
train_acc = 1
train_loss = .45
```

``` python
run.log({
    'accurac'
})
```
