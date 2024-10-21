---
title: "How can I change the directory my sweep logs to locally?"
tags: []
---

### How can I change the directory my sweep logs to locally?
You can change the path of the directory where W&B will log your run data by setting an environment variable `WANDB_DIR`. For example:

```python
os.environ["WANDB_DIR"] = os.path.abspath("your/directory")
```