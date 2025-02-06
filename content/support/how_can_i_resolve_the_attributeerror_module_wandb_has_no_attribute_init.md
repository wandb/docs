---
title: "How can I fix the error `AttributeError: module 'wandb' has no attribute 'init'`?"
toc_hide: true
type: docs
tags:
  - crashing and hanging runs
---

If you encounter the error `AttributeError: module 'wandb' has no attribute 'init'` when importing `wandb` in Python, `wandb` is not installed or the installation is corrupted, but a `wandb` directory exists in the current working directory. To fix this error, uninstall `wandb`, delete the directory, then install `wandb`:

```bash
pip uninstall wandb; rm -rI wandb; pip install wandb
```