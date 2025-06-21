---
title: controller()
object_type: python_sdk_actions
data_type_classification: function
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_sweep.py >}}




### <kbd>function</kbd> `controller`

```python
controller(
    sweep_id_or_config: Optional[str, Dict] = None,
    entity: Optional[str] = None,
    project: Optional[str] = None
) → _WandbController
```

Public sweep controller constructor. 



**Examples:**
 ```python
import wandb

tuner = wandb.controller(...)
print(tuner.sweep_config)
print(tuner.sweep_id)
tuner.configure_search(...)
tuner.configure_stopping(...)
``` 
