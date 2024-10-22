---
title: "How do I save code?‌"
tags:
   - artifacts
---
Use `save_code=True` in `wandb.init` to save the main script or notebook that launches the run. To save all code for a run, version the code with Artifacts. The following example demonstrates this process:

```python
code_artifact = wandb.Artifact(type="code")
code_artifact.add_file("./train.py")
wandb.log_artifact(code_artifact)
```