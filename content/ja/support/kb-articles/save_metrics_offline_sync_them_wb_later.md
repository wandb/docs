---
menu:
  support:
    identifier: ja-support-kb-articles-save_metrics_offline_sync_them_wb_later
support:
- experiments
- environment variables
- metrics
title: Is it possible to save metrics offline and sync them to W&B later?
toc_hide: true
type: docs
url: /support/:filename
---

By default, `wandb.init` starts a process that syncs metrics in real time to the cloud. For offline use, set two environment variables to enable offline mode and sync later.

Set the following environment variables:

1. `WANDB_API_KEY=$KEY`, where `$KEY` is the API Key from your [settings page](https://app.wandb.ai/settings).
2. `WANDB_MODE="offline"`.

Here is an example of implementing this in a script:

```python
import wandb
import os

os.environ["WANDB_API_KEY"] = "YOUR_KEY_HERE"
os.environ["WANDB_MODE"] = "offline"

config = {
    "dataset": "CIFAR10",
    "machine": "offline cluster",
    "model": "CNN",
    "learning_rate": 0.01,
    "batch_size": 128,
}

with wandb.init(project="offline-demo") as run:
    for i in range(100):
        run.log({"accuracy": i})
```

Sample terminal output is shown below:

{{< img src="/images/experiments/sample_terminal_output.png" alt="Offline mode terminal output" >}}

After completing work, run the following command to sync data to the cloud:

```shell
wandb sync wandb/dryrun-folder-name
```

{{< img src="/images/experiments/sample_terminal_output_cloud.png" alt="Cloud sync terminal output" >}}