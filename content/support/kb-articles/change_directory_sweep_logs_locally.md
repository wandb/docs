---
url: /support/:filename
title: "How can I change the directory my sweep logs to locally?"
toc_hide: true
type: docs
support:
   - sweeps
---
Set the logging directory for W&B run data by configuring the environment variable `WANDB_DIR`. For example:

```python
os.environ["WANDB_DIR"] = os.path.abspath("your/directory")
```