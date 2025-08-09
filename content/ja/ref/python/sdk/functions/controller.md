---
title: コントローラ()
data_type_classification: function
menu:
  reference:
    identifier: ja-ref-python-sdk-functions-controller
object_type: python_sdk_actions
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

公開 sweep コントローラのコンストラクタです。



**例:**
 ```python
import wandb

tuner = wandb.controller(...)
print(tuner.sweep_config)  # sweep の設定を表示
print(tuner.sweep_id)      # sweep の ID を表示
tuner.configure_search(...)  # 探索手法を設定
tuner.configure_stopping(...)  # 停止条件を設定
```