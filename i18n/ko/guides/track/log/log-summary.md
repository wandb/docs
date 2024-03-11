---
displayed_sidebar: default
---

# 로그 요약 메트릭

트레이닝 중 시간에 따라 변하는 값 외에도, 모델이나 전처리 단계를 요약하는 단일 값 추적이 종종 중요합니다. 이 정보를 W&B Run의 `summary` 사전에 로그하세요. Run의 요약 사전은 numpy 배열, PyTorch 텐서 또는 TensorFlow 텐서를 처리할 수 있습니다. 값이 이러한 유형 중 하나일 때, 전체 텐서를 이진 파일에 영구적으로 저장하고 요약 오브젝트에 최소값, 평균, 분산, 95번째 백분위수 등과 같은 고수준 메트릭을 저장합니다.

`wandb.log`로 로그된 마지막 값은 W&B Run의 요약 사전으로 자동 설정됩니다. 요약 메트릭 사전이 수정되면 이전 값은 사라집니다.

다음 코드조각은 W&B에 사용자 정의 요약 메트릭을 제공하는 방법을 보여줍니다:
```python
wandb.init(config=args)

best_accuracy = 0
for epoch in range(1, args.epochs + 1):
    test_loss, test_accuracy = test()
    if test_accuracy > best_accuracy:
        wandb.run.summary["best_accuracy"] = test_accuracy
        best_accuracy = test_accuracy
```

트레이닝이 완료된 후 기존 W&B Run의 요약 속성을 업데이트할 수 있습니다. 요약 속성을 업데이트하려면 [W&B Public API](../../../ref/python/public-api/README.md)를 사용하세요:

```python
api = wandb.Api()
run = api.run("username/project/run_id")
run.summary["tensor"] = np.random.random(1000)
run.summary.update()
```

## 요약 메트릭 사용자 정의하기

사용자 정의 요약 메트릭은 트레이닝의 마지막 단계가 아닌 최적의 단계에서 모델 성능을 포착하는 데 유용합니다. 예를 들어, 최종 값 대신 최대 정확도 또는 최소 손실 값과 같은 것을 포착하고 싶을 수 있습니다.

요약 메트릭은 `"min"`, `"max"`, `"mean"`, `"best"`, `"last"`, `"none"` 값을 받는 `summary` 인수를 사용하여 제어할 수 있습니다. `"best"` 파라미터는 선택적인 `objective` 인수와 함께 사용할 수 있으며, 이는 `"minimize"` 및 `"maximize"` 값이 허용됩니다. 다음은 기본 요약 행동(기록에서의 최종 값 사용) 대신 요약에서 손실의 최소값과 정확도의 최대값을 포착하는 예시입니다.

```python
import wandb
import random

random.seed(1)
wandb.init()
# 최소값에 관심이 있는 메트릭 정의
wandb.define_metric("loss", summary="min")
# 최대값에 관심이 있는 메트릭 정의
wandb.define_metric("acc", summary="max")
for i in range(10):
    log_dict = {
        "loss": random.uniform(0, 1 / (i + 1)),
        "acc": random.uniform(1 / (i + 1), 1),
    }
    wandb.log(log_dict)
```

다음은 프로젝트 페이지 워크스페이스의 사이드바에 고정된 열에서 결과적으로 나타나는 최소 및 최대 요약 값의 모습입니다:

![프로젝트 페이지 사이드바](/images/track/customize_sumary.png)