---
title: Customize log axes
displayed_sidebar: default
---

`define_metric`을 사용하여 **사용자 정의 x 축**을 설정하세요. 사용자 정의 x 축은 트레이닝 도중 과거의 다른 시간 단계에 비동기적으로 로그해야 할 때 유용합니다. 예를 들어, 에피소드별 보상과 단계별 보상을 추적해야 할 수 있는 RL에서 유용할 수 있습니다.

[Google Colab에서 `define_metric` 실습하기 →](http://wandb.me/define-metric-colab)

### 축 맞춤 설정

기본적으로 모든 메트릭은 동일한 x 축에 로그되며, 이는 W&B 내부 `step`입니다. 때로는 이전 단계에 로그하거나 다른 x 축을 사용하고 싶을 수도 있습니다.

다음은 기본 `step` 대신 사용자 정의 x 축 메트릭을 설정하는 예입니다.

```python
import wandb

wandb.init()
# 사용자 정의 x 축 메트릭 정의
wandb.define_metric("custom_step")
# 이 메트릭에 대해 플로팅될 메트릭 정의
wandb.define_metric("validation_loss", step_metric="custom_step")

for i in range(10):
    log_dict = {
        "train_loss": 1 / (i + 1),
        "custom_step": i**2,
        "validation_loss": 1 / (i + 1),
    }
    wandb.log(log_dict)
```

글롭을 사용하여 x 축을 설정할 수도 있습니다. 현재는 문자열 접두사를 가진 글롭만 사용할 수 있습니다. 다음 예제는 `"train/"` 접두사를 가진 모든 로그된 메트릭을 x 축 `"train/step"`에 플롯합니다:

```python
import wandb

wandb.init()
# 사용자 정의 x 축 메트릭 정의
wandb.define_metric("train/step")
# 다른 모든 train/ 메트릭이 이 단계 사용하도록 설정
wandb.define_metric("train/*", step_metric="train/step")

for i in range(10):
    log_dict = {
        "train/step": 2**i,  # 내부 W&B 단계에 따른 기하급수적 성장
        "train/loss": 1 / (i + 1),  # x 축은 train/step
        "train/accuracy": 1 - (1 / (1 + i)),  # x 축은 train/step
        "val/loss": 1 / (1 + i),  # x 축은 내부 wandb 단계
    }
    wandb.log(log_dict)
```