---
title: Ray 튠
description: W&B 를 Ray 튜닝과 연동하는 방법
menu:
  default:
    identifier: ko-guides-integrations-ray-tune
    parent: integrations
weight: 360
---

W&B 는 [Ray](https://github.com/ray-project/ray) 와 두 가지 간단한 인테그레이션을 제공합니다.

- `WandbLoggerCallback` 함수는 Tune 에서 보고된 메트릭을 자동으로 Wandb API 에 로그합니다.
- 함수 API 와 함께 사용할 수 있는 `setup_wandb()` 함수는 Tune 의 트레이닝 정보를 활용해 Wandb API 를 자동으로 초기화합니다. 평소처럼 Wandb API 를 사용할 수 있으며, 예를 들어 `run.log()` 를 통해 트레이닝 과정을 로그할 수 있습니다.

## 인테그레이션 설정하기

```python
from ray.air.integrations.wandb import WandbLoggerCallback
```

Wandb 설정은 `tune.run()` 의 config 파라미터에 wandb 키를 전달해서 진행합니다 (아래 예시 참고).

wandb config 항목의 값들은 `wandb.init()` 의 키워드 인수로 전달됩니다. 단, 아래 설정들은 `WandbLoggerCallback` 자체를 구성하는 데 사용됩니다:

### 파라미터

`project (str)`: Wandb 프로젝트 이름. 필수 항목입니다.

`api_key_file (str)`: Wandb API 키가 들어있는 파일 경로입니다.

`api_key (str)`: Wandb API 키입니다. `api_key_file` 대신 직접 입력할 수 있습니다.

`excludes (list)`: 로그에서 제외할 메트릭 리스트입니다.

`log_config (bool)`: 결과 딕셔너리의 config 파라미터를 로그로 남길지 여부입니다. 기본값은 False 입니다.

`upload_checkpoints (bool)`: True 로 설정하면 모델 체크포인트가 Artifacts 로 업로드됩니다. 기본값은 False 입니다.

### 예시

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

이 유틸리티 함수는 Ray Tune 과 함께 사용할 때 Wandb 를 초기화하는 데 도움을 줍니다. 기본적으로는 트레이닝 함수 내에서 `setup_wandb()` 를 호출하여 사용합니다:

```python
from ray.air.integrations.wandb import setup_wandb


def train_fn(config):
    # wandb 초기화
    wandb = setup_wandb(config)
    run = wandb.init(
        project=config["wandb"]["project"],
        api_key_file=config["wandb"]["api_key_file"],
    )

    for i in range(10):
        loss = config["a"] + config["b"]
        run.log({"loss": loss})
        tune.report(loss=loss)
    run.finish()


tuner = tune.Tuner(
    train_fn,
    param_space={
        # 탐색 공간을 여기서 정의합니다
        "a": tune.choice([1, 2, 3]),
        "b": tune.choice([4, 5, 6]),
        # wandb 설정
        "wandb": {"project": "Optimization_Project", "api_key_file": "/path/to/file"},
    },
)
results = tuner.fit()
```

## 예제 코드

인테그레이션이 어떻게 동작하는지 보여주는 예제를 아래에서 확인할 수 있습니다:

* [Colab](https://wandb.me/raytune-colab): 인테그레이션을 쉽게 체험할 수 있는 간단한 데모입니다.
* [Dashboard](https://wandb.ai/anmolmann/ray_tune): 예제에서 생성된 대시보드를 확인해 보세요.