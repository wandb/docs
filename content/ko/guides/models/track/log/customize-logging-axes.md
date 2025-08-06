---
title: 로그 축 커스터마이징
menu:
  default:
    identifier: ko-guides-models-track-log-customize-logging-axes
    parent: log-objects-and-media
---

W&B 에 메트릭을 로그할 때 커스텀 x축을 설정할 수 있습니다. 기본적으로 W&B 는 메트릭을 *step* 기준으로 로그합니다. 각 step 은 `wandb.Run.log()` API 호출에 해당합니다.

예를 들어, 아래 스크립트에는 10번 반복하는 `for` 루프가 있습니다. 각 반복마다 `validation_loss`라는 메트릭을 로그하며, step 번호가 매번 1씩 증가합니다.

```python
import wandb

with wandb.init() as run:
  # range 함수는 0부터 9까지의 숫자 시퀀스를 만듭니다.
  for i in range(10):
    log_dict = {
        "validation_loss": 1/(i+1)   
    }
    run.log(log_dict)
```

Projects 워크스페이스에서 `validation_loss` 메트릭은 x축을 `step` 으로 하여 표시되며, `wandb.Run.log()` 가 호출될 때마다 step 값이 1씩 증가합니다. 위 코드에서는 x축이 0, 1, 2, ..., 9 step 번호로 보여집니다.

{{< img src="/images/experiments/standard_axes.png" alt="x축을 step으로 사용하는 선그래프 패널." >}}

특정 상황에서는 로그 스케일의 x축과 같이 다른 축으로 메트릭을 표시하는 것이 더 유용할 수 있습니다. [`define_metric()`]({{< relref path="/ref/python/sdk/classes/run/#define_metric" lang="ko" >}}) 메소드를 사용하면 로그하는 모든 메트릭을 원하는 x축(커스텀 x축)으로 지정할 수 있습니다.

y축에 나타낼 메트릭은 `name` 파라미터에 지정하세요. `step_metric` 파라미터에 x축으로 사용할 메트릭을 넘기면 됩니다. 커스텀 메트릭을 로그할 때는, x축과 y축에 모두 값이 포함된 딕셔너리 형태로 key-value 쌍을 만들어주세요.

아래 예제를 참고하여 커스텀 x축 메트릭을 설정할 수 있습니다. `< >` 안의 값은 여러분의 값으로 바꿔서 사용하세요:

```python
import wandb

custom_step = "<custom_step>"  # 커스텀 x축 이름
metric_name = "<metric>"  # y축 메트릭 이름

with wandb.init() as run:
    # step_metric(x축)과 로그할 메트릭(y축) 지정
    run.define_metric(step_metric = custom_step, name = metric_name)

    for i in range(10):
        log_dict = {
            custom_step : int,  # x축 값
            metric_name : int,  # y축 값
        }
        run.log(log_dict)
```

아래 코드 예제에서는 `x_axis_squared`라는 커스텀 x축을 만듭니다. 이 x축의 값은 for 루프 인덱스 `i`의 제곱(`i**2`)입니다. y축에는 파이썬 내장 `random` 모듈을 이용해 임의의 `validation_loss` 값을 집어넣었습니다:

```python
import wandb
import random

with wandb.init() as run:
    run.define_metric(step_metric = "x_axis_squared", name = "validation_loss")

    for i in range(10):
        log_dict = {
            "x_axis_squared": i**2,
            "validation_loss": random.random(),
        }
        run.log(log_dict)
```

아래 이미지는 W&B App UI에서 만들어진 그래프 예시입니다. `validation_loss` 메트릭이 커스텀 x축 `x_axis_squared` 에 따라 그려지며, 이 값은 루프 인덱스 `i`의 제곱입니다. x축 값은 `0, 1, 4, 9, 16, 25, 36, 49, 64, 81`로서 각각 `0, 1, 2, ..., 9`의 제곱에 대응합니다.

{{< img src="/images/experiments/custom_x_axes.png" alt="커스텀 x축을 사용하는 선그래프 패널. W&B에는 루프 번호의 제곱으로 값이 로그됨." >}}

여러 메트릭에 대해 문자열 접두사를 활용한 `globs`를 이용해 커스텀 x축을 설정할 수도 있습니다. 예를 들어, 다음 코드 예시는 `train/*`로 시작하는 모든 메트릭을 x축 `train/step` 기준으로 표시합니다:

```python
import wandb

with wandb.init() as run:

    # 모든 train/ 메트릭을 해당 step 에 맞게 연결
    run.define_metric("train/*", step_metric="train/step")

    for i in range(10):
        log_dict = {
            "train/step": 2**i,  # W&B 내부 step 없이 지수적으로 증가
            "train/loss": 1 / (i + 1),  # x축은 train/step
            "train/accuracy": 1 - (1 / (1 + i)),  # x축은 train/step
            "val/loss": 1 / (1 + i),  # x축은 wandb 내부 step
        }
        run.log(log_dict)
```
