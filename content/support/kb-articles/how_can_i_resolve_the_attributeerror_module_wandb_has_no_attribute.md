---
url: /support/:filename
title: "How can I fix an error like `AttributeError: module 'wandb' has no attribute ...`?"
toc_hide: true
type: docs
support:
  - crashing and hanging runs
translationKey: how_can_i_resolve_the_attributeerror_module_wandb_has_no_attribute
---
If you encounter an error like `AttributeError: module 'wandb' has no attribute 'init'` or `AttributeError: module 'wandb' has no attribute 'login'` when importing `wandb` in Python, `wandb` is not installed or the installation is corrupted, but a `wandb` directory exists in the current working directory. To fix this error, uninstall `wandb`, delete the directory, then install `wandb`:

```bash
pip uninstall wandb; rm -rI wandb; pip install wandb
```