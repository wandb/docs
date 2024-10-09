---
title: Log summary metrics
displayed_sidebar: default
---

시간에 따라 트레이닝 중에 변경되는 값들 외에도, 모델이나 전처리 단계를 요약하는 단일 값을 추적하는 것이 중요할 때가 많습니다. 이 정보를 W&B Run의 `summary` 사전에 로그하세요. Run의 summary 사전은 numpy 배열, PyTorch 텐서 또는 TensorFlow 텐서를 처리할 수 있습니다. 이러한 유형 중 하나일 경우, 우리는 전체 텐서를 이진 파일에 저장하고 최소값, 평균, 분산, 95번째 백분위수 등의 높은 수준의 메트릭을 summary 오브젝트에 저장합니다.

`wandb.log`로 기록된 마지막 값은 자동으로 W&B Run의 summary 사전으로 설정됩니다. summary 메트릭 사전이 수정되면 이전 값은 사라집니다.

다음 코드조각은 W&B에 사용자 정의 summary 메트릭을 제공하는 방법을 보여줍니다:
```python
wandb.init(config=args)

best_accuracy = 0
for epoch in range(1, args.epochs + 1):
    test_loss, test_accuracy = test()
    if test_accuracy > best_accuracy:
        wandb.run.summary["best_accuracy"] = test_accuracy
        best_accuracy = test_accuracy
```

트레이닝이 완료된 후에 기존 W&B Run의 summary 속성을 업데이트할 수 있습니다. [W&B Public API](../../../ref/python/public-api/README.md)를 사용하여 summary 속성을 업데이트하세요:

```python
api = wandb.Api()
run = api.run("username/project/run_id")
run.summary["tensor"] = np.random.random(1000)
run.summary.update()
```

## summary 메트릭 사용자 지정하기

사용자 정의 메트릭 요약은 트레이닝의 마지막 단계가 아닌 최고의 단계에서 모델 성능을 캡처하는 데 유용합니다. 예를 들어, 최종 값 대신 최대 정확도나 최소 손실 값을 캡처하고 싶을 수 있습니다.

Summary 메트릭은 `"min"`, `"max"`, `"mean"`, `"best"`, `"last"`, `"none"` 값을 받아들이는 `define_metric`의 `summary` 인수를 사용하여 제어할 수 있습니다. `"best"` 파라미터는 `"minimize"` 및 `"maximize"` 값을 받아들이는 선택적인 `objective` 인수와 함께 사용할 수 있습니다. 다음은 기본 summary 행동 대신, 손실의 최소 값 및 정확도의 최대 값을 summary에 캡처하는 예시입니다.

```python
import wandb
import random

random.seed(1)
wandb.init()
# 최소값에 관심이 있는 메트릭을 정의합니다
wandb.define_metric("loss", summary="min")
# 최대값에 관심이 있는 메트릭을 정의합니다
wandb.define_metric("acc", summary="max")
for i in range(10):
    log_dict = {
        "loss": random.uniform(0, 1 / (i + 1)),
        "acc": random.uniform(1 / (i + 1), 1),
    }
    wandb.log(log_dict)
```

다음은 프로젝트 페이지 워크스페이스의 사이드바에 고정된 열에서 보이는 최소 및 최대 summary 값의 결과 모습입니다:

![Project Page Sidebar](/images/track/customize_sumary.png)