---
title: "How do I save code?‌"
tags:
   - artifacts
---

Use `save_code=True` in `wandb.init` to save the main script or notebook where you’re launching the run. To save all your code to a run, version code with Artifacts. Here’s an example:

```python
code_artifact = wandb.Artifact(type="code")
code_artifact.add_file("./train.py")
wandb.log_artifact(code_artifact)
```