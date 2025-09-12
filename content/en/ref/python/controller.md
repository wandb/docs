---
title: controller
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/wandb_sweep.py#L96-L120 >}}

Public sweep controller constructor.

```python
controller(
    sweep_id_or_config: Optional[Union[str, Dict]] = None,
    entity: Optional[str] = None,
    project: Optional[str] = None
) -> "_WandbController"
```

#### Examples:

```python
import wandb

tuner = wandb.controller(...)
print(tuner.sweep_config)
print(tuner.sweep_id)
tuner.configure_search(...)
tuner.configure_stopping(...)
```
