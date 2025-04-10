---
menu:
  support:
    identifier: ja-support-kb-articles-change_directory_sweep_logs_locally
support:
- sweeps
title: How can I change the directory my sweep logs to locally?
toc_hide: true
type: docs
url: /support/:filename
---

Set the logging directory for W&B run data by configuring the environment variable `WANDB_DIR`. For example:

```python
os.environ["WANDB_DIR"] = os.path.abspath("your/directory")
```