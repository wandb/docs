---
title: controller
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/e35e545afd28aab70ee9e2a9dcc5ec7cfb95b1a1/wandb/sdk/wandb_sweep.py#L95-L119 >}}

Public sweep controller constructor.

```python
controller(
    sweep_id_or_config: Optional[Union[str, Dict]] = None,
    entity: Optional[str] = None,
    project: Optional[str] = None
) -> "_WandbController"
```

#### Usage:

```python
import wandb

tuner = wandb.controller(...)
print(tuner.sweep_config)
print(tuner.sweep_id)
tuner.configure_search(...)
tuner.configure_stopping(...)
```
