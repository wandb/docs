---
title: controller
menu:
  reference:
    identifier: ja-ref-python-controller
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_sweep.py#L95-L119 >}}

パブリック sweep コントローラのコンストラクタ。

```python
controller(
    sweep_id_or_config: Optional[Union[str, Dict]] = None,
    entity: Optional[str] = None,
    project: Optional[str] = None
) -> "_WandbController"
```

#### 使用方法:

```python
import wandb

tuner = wandb.controller(...)
print(tuner.sweep_config)
print(tuner.sweep_id)
tuner.configure_search(...)
tuner.configure_stopping(...)
```
