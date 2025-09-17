---
title: controller()
data_type_classification: function
menu:
  reference:
    identifier: ja-ref-python-sdk-functions-controller
object_type: python_sdk_actions
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_sweep.py >}}




### <kbd>関数</kbd> `controller`

```python
controller(
    sweep_id_or_config: Optional[str, Dict] = None,
    entity: Optional[str] = None,
    project: Optional[str] = None
) → _WandbController
```

sweep コントローラの公開コンストラクタです。 



**例:**
 ```python
import wandb

tuner = wandb.controller(...)
print(tuner.sweep_config)
print(tuner.sweep_id)
tuner.configure_search(...)
tuner.configure_stopping(...)
```