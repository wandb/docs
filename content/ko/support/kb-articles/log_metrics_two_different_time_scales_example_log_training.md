---
title: Can I log metrics on two different time scales?
menu:
  support:
    identifier: ko-support-kb-articles-log_metrics_two_different_time_scales_example_log_training
support:
- experiments
- metrics
toc_hide: true
type: docs
url: /ko/support/:filename
---

예를 들어, 배치 당 트레이닝 정확도 와 에포크 당 검증 정확도 를 로그하고 싶다고 가정해 보겠습니다.

예, 메트릭 과 함께 `batch` 및 `epoch`와 같은 인덱스를 로그합니다. 한 단계에서 `wandb.log({'train_accuracy': 0.9, 'batch': 200})`을 사용하고 다른 단계에서 `wandb.log({'val_accuracy': 0.8, 'epoch': 4})`을 사용합니다. UI에서 원하는 값을 각 차트의 x축으로 설정합니다. 특정 인덱스에 대한 기본 x축을 설정하려면 [Run.define_metric()]({{< relref path="/ref/python/run.md#define_metric" lang="ko" >}})을 사용하세요. 제공된 예제의 경우 다음 코드를 사용하세요.

```python
wandb.init()

wandb.define_metric("batch")
wandb.define_metric("epoch")

wandb.define_metric("train_accuracy", step_metric="batch")
wandb.define_metric("val_accuracy", step_metric="epoch")
```
