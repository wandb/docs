---
displayed_sidebar: default
---

# 로그 축 사용자 정의하기

`define_metric`을 사용하여 **사용자 정의 x 축**을 설정하세요. 사용자 정의 x 축은 트레이닝 중에 과거의 다른 시간 단계로 비동기적으로 로그를 기록해야 할 때 유용합니다. 예를 들어, RL에서 에피소드 별 보상과 단계별 보상을 추적해야 할 때 유용할 수 있습니다.

[Google Colab에서 `define_metric` 실습해보기 →](http://wandb.me/define-metric-colab)

### 축 사용자 정의하기

기본적으로 모든 메트릭은 W&B 내부의 `step`이라는 동일한 x 축에 대해 로그됩니다. 때때로, 이전 단계로 로그하거나 다른 x 축을 사용하고 싶을 수 있습니다.

여기 기본 단계 대신 사용자 정의 x 축 메트릭을 설정하는 예가 있습니다.

```python
import wandb

wandb.init()
# 우리의 사용자 정의 x 축 메트릭 정의하기
wandb.define_metric("custom_step")
# 어떤 메트릭이 그것에 대해 플롯될지 정의하기
wandb.define_metric("validation_loss", step_metric="custom_step")

for i in range(10):
    log_dict = {
        "train_loss": 1 / (i + 1),
        "custom_step": i**2,
        "validation_loss": 1 / (i + 1),
    }
    wandb.log(log_dict)
```

x 축은 글로브를 사용하여 설정할 수도 있습니다. 현재는 문자열 접두어를 가진 글로브만 사용할 수 있습니다. 다음 예제는 접두어 `"train/"`을 가진 모든 로그된 메트릭을 x 축 `"train/step"`에 플롯합니다:

```python
import wandb

wandb.init()
# 우리의 사용자 정의 x 축 메트릭 정의하기
wandb.define_metric("train/step")
# 모든 다른 train/ 메트릭들이 이 단계를 사용하도록 설정하기
wandb.define_metric("train/*", step_metric="train/step")

for i in range(10):
    log_dict = {
        "train/step": 2**i,  # 내부 W&B 단계와 함께하는 지수적 성장
        "train/loss": 1 / (i + 1),  # x 축은 train/step
        "train/accuracy": 1 - (1 / (1 + i)),  # x 축은 train/step
        "val/loss": 1 / (1 + i),  # x 축은 내부 wandb 단계
    }
    wandb.log(log_dict)
```