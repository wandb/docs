---
title: Ray Tune
description: Ray Tune과 W&B를 통합하는 방법
menu:
  default:
    identifier: ko-guides-integrations-ray-tune
    parent: integrations
weight: 360
---

W&B는 두 가지 경량 인테그레이션을 제공하여 [Ray](https://github.com/ray-project/ray)와 통합됩니다.

- `WandbLoggerCallback` 함수는 Tune에 보고된 메트릭을 Wandb API에 자동으로 로그합니다.
- 함수 API와 함께 사용할 수 있는 `setup_wandb()` 함수는 Tune의 트레이닝 정보로 Wandb API를 자동으로 초기화합니다. `wandb.log()`를 사용하여 트레이닝 프로세스를 기록하는 등, 평소와 같이 Wandb API를 사용할 수 있습니다.

## 인테그레이션 설정

```python
from ray.air.integrations.wandb import WandbLoggerCallback
```

Wandb 설정은 `tune.run()`의 config 파라미터에 wandb 키를 전달하여 수행됩니다 (아래 예제 참조).

wandb config 항목의 내용은 `wandb.init()`에 키워드 인수로 전달됩니다. 예외는 `WandbLoggerCallback` 자체를 구성하는 데 사용되는 다음 설정입니다.

### 파라미터

`project (str)`: Wandb 프로젝트 이름. 필수 항목입니다.

`api_key_file (str)`: Wandb API 키가 포함된 파일의 경로입니다.

`api_key (str)`: Wandb API 키. `api_key_file` 설정의 대안입니다.

`excludes (list)`: 로그에서 제외할 메트릭 목록입니다.

`log_config (bool)`: results 사전의 config 파라미터를 기록할지 여부입니다. 기본값은 False입니다.

`upload_checkpoints (bool)`: True이면 모델 체크포인트가 Artifacts로 업로드됩니다. 기본값은 False입니다.

### 예제

```python
from ray import tune, train
from ray.air.integrations.wandb import WandbLoggerCallback


def train_fc(config):
    for i in range(10):
        train.report({"mean_accuracy": (i + config["alpha"]) / 10})


tuner = tune.Tuner(
    train_fc,
    param_space={
        "alpha": tune.grid_search([0.1, 0.2, 0.3]),
        "beta": tune.uniform(0.5, 1.0),
    },
    run_config=train.RunConfig(
        callbacks=[
            WandbLoggerCallback(
                project="<your-project>", api_key="<your-api-key>", log_config=True
            )
        ]
    ),
)

results = tuner.fit()
```

## setup_wandb

```python
from ray.air.integrations.wandb import setup_wandb
```

이 유틸리티 함수는 Ray Tune과 함께 사용하기 위해 Wandb를 초기화하는 데 도움이 됩니다. 기본 사용법은 트레이닝 함수에서 `setup_wandb()`를 호출하는 것입니다.

```python
from ray.air.integrations.wandb import setup_wandb


def train_fn(config):
    # Initialize wandb
    wandb = setup_wandb(config)

    for i in range(10):
        loss = config["a"] + config["b"]
        wandb.log({"loss": loss})
        tune.report(loss=loss)


tuner = tune.Tuner(
    train_fn,
    param_space={
        # define search space here
        "a": tune.choice([1, 2, 3]),
        "b": tune.choice([4, 5, 6]),
        # wandb configuration
        "wandb": {"project": "Optimization_Project", "api_key_file": "/path/to/file"},
    },
)
results = tuner.fit()
```

## 예제 코드

인테그레이션 작동 방식을 보여주는 몇 가지 예제를 만들었습니다.

* [Colab](http://wandb.me/raytune-colab): 인테그레이션을 시도해 볼 수 있는 간단한 데모입니다.
* [Dashboard](https://wandb.ai/anmolmann/ray_tune): 예제에서 생성된 대시보드를 봅니다.
