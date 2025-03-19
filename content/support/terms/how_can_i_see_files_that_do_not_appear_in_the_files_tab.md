---
title: How can I see files that do not appear in the Files tab?
toc_hide: true
type: docs
support:
  - experiments
---

The Files tab shows a maximum of 10,000 files. To download all files, use the [public API]({{< relref "/ref/python/public-api/api.md" >}}):

```python
import wandb

api = wandb.Api()
run = api.run('<entity>/<project>/<run_id>')
run.file('<file>').download()

for f in run.files():
    if <condition>:
        f.download()
```