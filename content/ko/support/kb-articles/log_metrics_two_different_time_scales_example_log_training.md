---
title: 두 가지 다른 시간 척도에서 메트릭을 로그할 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-log_metrics_two_different_time_scales_example_log_training
support:
- 실험
- 메트릭
toc_hide: true
type: docs
url: /support/:filename
---

예를 들어, 저는 배치마다 트레이닝 정확도, 에포크마다 검증 정확도를 로그로 남기고 싶습니다.

네, `batch`와 `epoch` 같은 인덱스를 메트릭과 함께 로그하세요. 한 단계에서는 `wandb.Run.log()({'train_accuracy': 0.9, 'batch': 200})`를, 또 다른 단계에서는 `wandb.Run.log()({'val_accuracy': 0.8, 'epoch': 4})`를 사용하면 됩니다. UI에서 원하는 값을 각 차트의 x축으로 설정하세요. 특정 인덱스에 대해 기본 x축을 지정하려면 [Run.define_metric()]({{< relref path="/ref/python/sdk/classes/run#define_metric" lang="ko" >}})을 사용하세요. 아래는 예시에 해당하는 코드입니다:

```python
import wandb

with wandb.init() as run:
   run.define_metric("batch")
   run.define_metric("epoch")

   run.define_metric("train_accuracy", step_metric="batch")
   run.define_metric("val_accuracy", step_metric="epoch")
```