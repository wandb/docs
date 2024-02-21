---
displayed_sidebar: default
---

# 요약 메트릭 로그하기

학습하는 동안 시간에 따라 변하는 값 외에도 모델이나 전처리 단계를 요약하는 단일 값을 추적하는 것이 종종 중요합니다. 이 정보를 W&B Run의 `summary` 사전에 로그하세요. Run의 요약 사전은 numpy 배열, PyTorch 텐서 또는 TensorFlow 텐서를 처리할 수 있습니다. 값이 이러한 유형 중 하나인 경우 전체 텐서를 이진 파일에 유지하고 요약 객체에 최소값, 평균, 분산, 95번째 백분위수 등과 같은 고급 메트릭을 저장합니다.

`wandb.log`로 로그된 마지막 값이 W&B Run의 요약 사전으로 자동 설정됩니다. 요약 메트릭 사전이 수정되면 이전 값은 사라집니다.

다음 코드 조각은 W&B에 사용자 정의 요약 메트릭을 제공하는 방법을 보여줍니다:
```python
wandb.init(config=args)

best_accuracy = 0
for epoch in range(1, args.epochs + 1):
    test_loss, test_accuracy = test()
    if test_accuracy > best_accuracy:
        wandb.run.summary["best_accuracy"] = test_accuracy
        best_accuracy = test_accuracy
```

학습이 완료된 후 기존 W&B Run의 요약 속성을 업데이트할 수 있습니다. 요약 속성을 업데이트하려면 [W&B Public API](../../../ref/python/public-api/README.md)를 사용하세요:

```python
api = wandb.Api()
run = api.run("username/project/run_id")
run.summary["tensor"] = np.random.random(1000)
run.summary.update()
```

## 요약 메트릭 사용자 정의하기

사용자 정의 요약 메트릭은 학습의 마지막 단계가 아닌 최고 단계에서 모델 성능을 캡처하는 데 유용합니다. 예를 들어, 최종 값 대신 최대 정확도 또는 최소 손실 값을 캡처하고자 할 수 있습니다.

요약 메트릭은 `"min"`, `"max"`, `"mean"`, `"best"`, `"last"`, `"none"` 값을 허용하는 `summary` 인수를 사용하여 제어할 수 있습니다. `"best"` 파라미터는 선택적 `objective` 인수와 함께 사용될 수 있으며 `"minimize"`와 `"maximize"` 값을 허용합니다. 다음은 기본 요약 동작이 역사에서 최종 값을 사용하는 대신 요약에서 손실 최소값과 정확도 최대값을 캡처하는 예입니다.

```python
import wandb
import random

random.seed(1)
wandb.init()
# 관심 있는 최소 메트릭 정의하기
wandb.define_metric("loss", summary="min")
# 관심 있는 최대 메트릭 정의하기
wandb.define_metric("acc", summary="max")
for i in range(10):
    log_dict = {
        "loss": random.uniform(0, 1 / (i + 1)),
        "acc": random.uniform(1 / (i + 1), 1),
    }
    wandb.log(log_dict)
```

프로젝트 페이지 워크스페이스의 사이드바에 고정된 열로 표시된 결과 최소 및 최대 요약 값은 다음과 같습니다:

![프로젝트 페이지 사이드바](/images/track/customize_summary.png)