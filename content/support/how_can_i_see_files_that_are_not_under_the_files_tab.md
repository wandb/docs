---
title: How can I see files that do not appear in the Files tab?
toc_hide: true
type: docs
tags:
  - experiments
---

The Files tab is limited to 10,000 files. Use the Public API to download hidden files:

```python
import wandb

api = wandb.Api()
run = api.run('<entity>/<project>/<run_id>')
run.file('<file>').download()

for f in run.files():
    if <condition>:
        f.download()
```