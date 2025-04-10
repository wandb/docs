---
title: What if I want to log some metrics on batches and some metrics only on epochs?
menu:
  support:
    identifier: ko-support-kb-articles-log_metrics_batches_some_metrics_epochs
support:
- experiments
- metrics
toc_hide: true
type: docs
url: /support/:filename
---

각 배치에서 특정 메트릭을 로그하고 플롯을 표준화하려면 원하는 x축 값과 함께 메트릭을 로그하세요. 사용자 정의 플롯에서 편집을 클릭하고 사용자 정의 x축을 선택합니다.

```python
wandb.log({"batch": batch_idx, "loss": 0.3})
wandb.log({"epoch": epoch, "val_acc": 0.94})
```
