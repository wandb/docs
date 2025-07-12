---
url: /support/:filename
title: "How do I programmatically access the human-readable run name?"
toc_hide: true
type: docs
support:
   - experiments
---
The `.name` attribute of a [`wandb.Run`]({{< relref "/ref/python/sdk/classes/run" >}}) is accessible as follows:

```python
import wandb

wandb.init()
run_name = wandb.run.name
```