---
url: /support/:filename
title: "Can I just set the run name to the run ID?"
toc_hide: true
type: docs
support:
   - experiments
---

Yes. To overwrite the run name with the run ID, use the following code snippet:

```python
import wandb

with wandb.init() as run:
   run.name = run.id
   run.save()
```