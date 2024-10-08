---
title: Ray Tune
description: W&B를 Ray Tune과 통합하는 방법.
slug: /guides/integrations/ray-tune
displayed_sidebar: default
---

W&B는 두 가지 가벼운 인테그레이션을 제공하여 [Ray](https://github.com/ray-project/ray)와 통합됩니다.

하나는 `WandbLoggerCallback`으로, wandb API에 로그 메트릭을 자동으로 튜닝합니다. 다른 하나는 `@wandb_mixin` 데코레이터로, 함수 API와 함께 사용할 수 있습니다. 이것은 Tune의 트레이닝 정보를 사용하여 wandb API를 자동으로 초기화합니다. wandb.log()를 사용하여 트레이닝 프로세스를 로그하듯이 wandb API를 일반적으로 사용할 수 있습니다.

## WandbLoggerCallback

```python
from ray.air.integrations.wandb import WandbLoggerCallback
```

wandb 설정은 `tune.run()`의 설정 파라미터에 wandb 키를 전달하여 이루어집니다 (아래 예 참조).

wandb 설정 항목의 내용은 인수로 `wandb.init()`에 전달됩니다. 예외는 다음의 설정으로, `WandbLoggerCallback` 자체를 설정하는 데 사용됩니다:

### 파라미터

`api_key_file (str)` – Wandb `API KEY`가 포함된 파일의 경로.

`api_key (str)` – Wandb API 키. `api_key_file`을 설정하는 대안.

`excludes (list)` – `log`에서 제외되어야 하는 메트릭 목록.

`log_config (bool)` – 결과 딕셔너리의 설정 파라미터가 로그되어야 하는지 나타내는 부울 값입니다. 이는 `PopulationBasedTraining`에서처럼 파라미터가 트레이닝 동안 변경될 경우 의미가 있습니다. 기본값은 False입니다.

### 예시

```python
from ray import tune, train
from ray.tune.logger import DEFAULT_LOGGERS
from ray.air.integrations.wandb import WandbLoggerCallback

def train_fc(config):
    for i in range(10):
        train.report({"mean_accuracy":(i + config['alpha']) / 10})

search_space = {
    'alpha': tune.grid_search([0.1, 0.2, 0.3]),
    'beta': tune.uniform(0.5, 1.0)
}

analysis = tune.run(
    train_fc,
    config=search_space,
    callbacks=[WandbLoggerCallback(
        project="<your-project>",
        api_key="<your-name>",
        log_config=True
    )]
)

best_trial = analysis.get_best_trial("mean_accuracy", "max", "last")
```

## wandb_mixin

```python
ray.tune.integration.wandb.wandb_mixin(func)
```

이 Ray Tune Trainable `mixin`은 `Trainable` 클래스와 함수 API의 `@wandb_mixin`을 사용하는 데 wandb API를 초기화하는 데 도움을 줍니다.

기본 사용법으로, 트레이닝 함수의 앞에 `@wandb_mixin` 데코레이터를 붙이기만 하면 됩니다:

```python
from ray.tune.integration.wandb import wandb_mixin


@wandb_mixin
def train_fn(config):
    wandb.log()
```

wandb 설정은 `tune.run()`의 `config` 파라미터에 `wandb key`를 전달하여 이루어집니다 (아래 예 참조).

wandb 설정 항목의 내용은 인수로 `wandb.init()`에 전달됩니다. 예외는 다음의 설정으로, `WandbTrainableMixin` 자체를 설정하는 데 사용됩니다:

### 파라미터

`api_key_file (str)` – Wandb `API KEY`가 포함된 파일의 경로.

`api_key (str)` – Wandb API 키. `api_key_file`을 설정하는 대안.

Wandb의 `group`, `run_id` 및 `run_name`은 Tune에 의해 자동으로 선택되지만, 해당 설정 값을 채워 넣어 덮어쓸 수 있습니다.

기타 유효한 설정에 대한 자세한 내용은 [`init()` reference](/ref/python/init/)를 참조하십시오.

### 예시:

```python
from ray import tune
from ray.tune.integration.wandb import wandb_mixin


@wandb_mixin
def train_fn(config):
    for i in range(10):
        loss = self.config["a"] + self.config["b"]
        wandb.log({"loss": loss})
        tune.report(loss=loss)


tune.run(
    train_fn,
    config={
        # 여기에서 탐색 공간 정의
        "a": tune.choice([1, 2, 3]),
        "b": tune.choice([4, 5, 6]),
        # wandb 설정
        "wandb": {"project": "Optimization_Project", "api_key_file": "/path/to/file"},
    },
)
```

## 예제 코드

인테그레이션이 작동하는 방식을 볼 수 있도록 몇 가지 예를 만들었습니다:

* [Colab](http://wandb.me/raytune-colab): 인테그레이션을 시도해 볼 수 있는 간단한 데모.
* [Dashboard](https://wandb.ai/anmolmann/ray_tune): 예제에서 생성된 대시보드 보기.