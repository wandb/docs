---
title: "How can I fix the error `AttributeError: module 'wandb' has no attribute 'login'`?"
toc_hide: true
type: docs
tags:
  - crashing and hanging runs
---

If you encounter the error `AttributeError: module 'wandb' has no attribute 'login'` when importing `wandb` in Python, 
 error might appear when trying to import `wandb` from a Python environment, `wandb` is not installed or the installation is corrupted,  but a `wandb` directory exists in the current working directory.  To fix this error, uninstall `wandb`, delete the directory, then install `wandb`:

```bash
pip uninstall wandb && pip install wandb
```