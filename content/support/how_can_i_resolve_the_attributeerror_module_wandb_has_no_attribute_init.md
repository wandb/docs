---
title: "How can I fix the error `AttributeError: module 'wandb' has no attribute 'init'`?"
toc_hide: true
type: docs
tags:
  - crashing and hanging runs
---

This error might appear when trying to import `wandb` from a Python environment that does not have `wandb` installed, but there is a directory named `wandb` where the code is being executed. We recommend:

```bash
pip uninstall wandb && pip install wandb
```