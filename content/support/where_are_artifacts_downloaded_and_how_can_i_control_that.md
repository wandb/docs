---
title: Where are artifacts downloaded, and how can I control that?
toc_hide: true
type: docs
tags:
  - artifacts
  - environment variables
---

By default, artifacts download to the `artifacts/` folder. You can change this by:

```python
wandb.Artifact().download(root="<path_to_download>")
```

Or by setting an environment variable:

```python
import os
os.environ["WANDB_ARTIFACT_DIR"] = "<path_to_download>"
```