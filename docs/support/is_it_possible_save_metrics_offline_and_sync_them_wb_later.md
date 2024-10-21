---
title: "Is it possible to save metrics offline and sync them to W&B later?"
tags: []
---

### Is it possible to save metrics offline and sync them to W&B later?
By default, `wandb.init` starts a process that syncs metrics in real time to our cloud hosted app. If your machine is offline, you don't have internet access, or you just want to hold off on the upload, here's how to run `wandb` in offline mode and sync later.

You will need to set two [environment variables](./environment-variables.md).

1. `WANDB_API_KEY=$KEY`, where `$KEY` is the API Key from your [settings page](https://app.wandb.ai/settings)
2. `WANDB_MODE="offline"`

And here's a sample of what this would look like in your script:

```python
import wandb
import os

os.environ["WANDB_API_KEY"] = YOUR_KEY_HERE
os.environ["WANDB_MODE"] = "offline"

config = {
    "dataset": "CIFAR10",
    "machine": "offline cluster",
    "model": "CNN",
    "learning_rate": 0.01,
    "batch_size": 128,
}

wandb.init(project="offline-demo")

for i in range(100):
    wandb.log({"accuracy": i})
```

Here's a sample terminal output:

![](/images/experiments/sample_terminal_output.png)

And once you're ready, just run a sync command to send that folder to the cloud.

```shell
wandb sync wandb/dryrun-folder-name
```

![](/images/experiments/sample_terminal_output_cloud.png)