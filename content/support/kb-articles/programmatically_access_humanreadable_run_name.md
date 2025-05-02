---
url: /support/:filename
title: "How do I programmatically access the human-readable run name?"
toc_hide: true
type: docs
support:
   - experiments
translationKey: programmatically_access_humanreadable_run_name
---
The `.name` attribute of a [`wandb.Run`]({{< relref "/ref/python/run.md" >}}) is accessible as follows:

```python
import wandb

wandb.init()
run_name = wandb.run.name
```