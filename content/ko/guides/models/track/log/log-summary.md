---
title: Log summary metrics
menu:
  default:
    identifier: ko-guides-models-track-log-log-summary
    parent: log-objects-and-media
---

트레이닝 과정에서 시간이 지남에 따라 변하는 값 외에도 모델 또는 전처리 단계를 요약하는 단일 값을 추적하는 것이 중요합니다. 이 정보를 W&B Run의 `summary` 사전에 기록하세요. Run의 요약 사전은 numpy 배열, PyTorch 텐서 또는 TensorFlow 텐서를 처리할 수 있습니다. 값이 이러한 유형 중 하나인 경우 전체 텐서를 이진 파일로 유지하고 min, mean, variance, 백분위수 등과 같은 높은 수준의 메트릭을 요약 오브젝트에 저장합니다.

`wandb.log`로 기록된 마지막 값은 W&B Run에서 자동으로 요약 사전으로 설정됩니다. 요약 메트릭 사전이 수정되면 이전 값은 손실됩니다.

다음 코드 조각은 W&B에 사용자 정의 요약 메트릭을 제공하는 방법을 보여줍니다.

```python
wandb.init(config=args)

best_accuracy = 0
for epoch in range(1, args.epochs + 1):
    test_loss, test_accuracy = test()
    if test_accuracy > best_accuracy:
        wandb.run.summary["best_accuracy"] = test_accuracy
        best_accuracy = test_accuracy
```

트레이닝이 완료된 후 기존 W&B Run의 요약 속성을 업데이트할 수 있습니다. [W&B Public API]({{< relref path="/ref/python/public-api/" lang="ko" >}})를 사용하여 요약 속성을 업데이트하세요.

```python
api = wandb.Api()
run = api.run("username/project/run_id")
run.summary["tensor"] = np.random.random(1000)
run.summary.update()
```

## 요약 메트릭 사용자 정의

사용자 정의 메트릭 요약은 `wandb.summary`에서 트레이닝의 마지막 단계 대신 최상의 단계에서 모델 성능을 캡처하는 데 유용합니다. 예를 들어 최종 값 대신 최대 정확도 또는 최소 손실 값을 캡처할 수 있습니다.

요약 메트릭은 `define_metric`의 `summary` 인수를 사용하여 제어할 수 있습니다. 이 인수는 `min`, `max`, `mean`, `best`, `last` 및 `none` 값을 허용합니다. `best` 파라미터는 `minimize` 및 `maximize` 값을 허용하는 선택적 `objective` 인수와 함께만 사용할 수 있습니다. 다음은 기록의 최종 값을 사용하는 기본 요약 동작 대신 요약에서 손실의 가장 낮은 값과 정확도의 최대값을 캡처하는 예입니다.

```python
import wandb
import random

random.seed(1)
wandb.init()
# define a metric we are interested in the minimum of
wandb.define_metric("loss", summary="min")
# define a metric we are interested in the maximum of
wandb.define_metric("acc", summary="max")
for i in range(10):
    log_dict = {
        "loss": random.uniform(0, 1 / (i + 1)),
        "acc": random.uniform(1 / (i + 1), 1),
    }
    wandb.log(log_dict)
```

다음은 Project Page 워크스페이스의 사이드바에 고정된 열에 표시되는 최소 및 최대 요약 값의 모습입니다.

{{< img src="/images/track/customize_sumary.png" alt="Project Page Sidebar" >}}
