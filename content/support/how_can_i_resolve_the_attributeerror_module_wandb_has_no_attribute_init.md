---
title: "How can I fix the error `AttributeError: module 'wandb' has no attribute 'init'`?"
toc_hide: true
type: docs
tags:
  - crashing and hanging runs
---

If you encounter the error `AttributeError: module 'wandb' has no attribute 'init'` when importing the `wandb` Python library, the library is not installed or the installation is corrupted, but a `wandb` directory exists in the current working directory. To fix this error, uninstall the `wandb` library, delete the directory, then install the library again:

```bash
pip uninstall wandb; rm -rI wandb; pip install wandb
```