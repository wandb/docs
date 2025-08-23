---
title: 배치마다 일부 메트릭을 로그하고, 에포크마다만 다른 메트릭을 로그하고 싶으면 어떻게 하나요?
menu:
  support:
    identifier: ko-support-kb-articles-log_metrics_batches_some_metrics_epochs
support:
- 실험
- 메트릭
toc_hide: true
type: docs
url: /support/:filename
---

각 배치에서 특정 메트릭을 로그하고 플롯을 표준화하려면, 메트릭과 함께 원하는 x축 값을 로그하세요. 커스텀 플롯에서 편집을 클릭한 후 원하는 x축을 선택할 수 있습니다.

```python
import wandb

with wandb.init() as run:
    # 배치별로 로그
    run.log({"batch": batch_idx, "loss": 0.3})
    # 에포크별로 로그
    run.log({"epoch": epoch, "val_acc": 0.94})
```