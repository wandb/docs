---
url: /support/:filename
title: "My run's state is `crashed` on the UI but is still running on my machine. What do I do to get my data back?"
toc_hide: true
type: docs
support:
   - experiments
---
You likely lost connection to your machine during training. Recover data by running [`wandb sync [PATH_TO_RUN]`]({{< relref "/ref/cli/wandb-sync.md" >}}). The path to your run is a folder in your `wandb` directory that matches the Run ID of the ongoing run.