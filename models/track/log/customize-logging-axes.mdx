---
title: Customize log axes
menu:
  default:
    identifier: ko-guides-models-track-log-customize-logging-axes
    parent: log-objects-and-media
---

`define_metric`을 사용하여 **사용자 정의 x축**을 설정하세요. 사용자 정의 x축은 트레이닝 중 과거의 다른 타임 스텝에 비동기적으로 로그해야 하는 상황에서 유용합니다. 예를 들어, 에피소드별 보상과 스텝별 보상을 추적할 수 있는 RL에서 유용할 수 있습니다.

[Google Colab에서 `define_metric`을 직접 사용해 보세요 →](http://wandb.me/define-metric-colab)

### 축 사용자 정의

기본적으로 모든 메트릭은 W&B 내부 `step`인 동일한 x축에 대해 기록됩니다. 때로는 이전 스텝에 로그하거나 다른 x축을 사용하고 싶을 수 있습니다.

다음은 기본 스텝 대신 사용자 정의 x축 메트릭을 설정하는 예입니다.

```python
import wandb

wandb.init()
# 사용자 정의 x축 메트릭 정의
wandb.define_metric("custom_step")
# 어떤 메트릭을 기준으로 플롯할지 정의
wandb.define_metric("validation_loss", step_metric="custom_step")

for i in range(10):
    log_dict = {
        "train_loss": 1 / (i + 1),
        "custom_step": i**2,
        "validation_loss": 1 / (i + 1),
    }
    wandb.log(log_dict)
```

x축은 glob을 사용하여 설정할 수도 있습니다. 현재 문자열 접두사가 있는 glob만 사용할 수 있습니다. 다음 예제는 접두사 `"train/"`가 있는 기록된 모든 메트릭을 x축 `"train/step"`에 플롯합니다.

```python
import wandb

wandb.init()
# 사용자 정의 x축 메트릭 정의
wandb.define_metric("train/step")
# 다른 모든 train/ 메트릭이 이 스텝을 사용하도록 설정
wandb.define_metric("train/*", step_metric="train/step")

for i in range(10):
    log_dict = {
        "train/step": 2**i,  # 내부 W&B 스텝으로 지수적 증가
        "train/loss": 1 / (i + 1),  # x축은 train/step
        "train/accuracy": 1 - (1 / (1 + i)),  # x축은 train/step
        "val/loss": 1 / (1 + i),  # x축은 내부 wandb step
    }
    wandb.log(log_dict)
```