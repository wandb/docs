---
description: How to integrate W&B with Ray Tune.
slug: /guides/integrations/ray-tune
displayed_sidebar: default
---

# Ray Tune

W&B는 두 가지 경량 인테그레이션을 제공하여 [Ray](https://github.com/ray-project/ray)와 통합합니다.

하나는 `WandbLoggerCallback`로, Tune에 보고된 메트릭을 Wandb API에 자동으로 로그합니다. 다른 하나는 함수 API와 함께 사용할 수 있는 `@wandb_mixin` 데코레이터로, Tune의 트레이닝 정보로 Wandb API를 자동으로 초기화합니다. `wandb.log()`를 사용하여 트레이닝 프로세스를 로그하는 것과 같이 평소처럼 Wandb API를 사용할 수 있습니다.

## WandbLoggerCallback

```python
from ray.air.integrations.wandb import WandbLoggerCallback
```

Wandb 설정은 `tune.run()`의 config 파라미터에 wandb 키를 전달하여 수행됩니다(아래 예제 참조).

wandb config 항목의 내용은 키워드 인수로 `wandb.init()`에 전달됩니다. 다음 설정을 제외하고는, 이 설정들은 `WandbLoggerCallback` 자체를 설정하는데 사용됩니다:

### 파라미터

`api_key_file (str)` – `Wandb API 키`를 포함한 파일의 경로.

`api_key (str)` – Wandb API 키. `api_key_file` 설정의 대안.

`excludes (list)` – 로그에서 제외해야 할 메트릭의 리스트.

`log_config (bool)` – 결과 dict의 config 파라미터가 로그되어야 하는지를 나타내는 불리언. 예를 들어 `PopulationBasedTraining`과 같이 트레이닝 중에 파라미터가 변경될 경우 의미가 있습니다. 기본값은 False입니다.

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

## wandb\_mixin

```python
ray.tune.integration.wandb.wandb_mixin(func)
```

이 Ray Tune Trainable `mixin`은 `Trainable` 클래스 또는 함수 API에 대한 `@wandb_mixin`과 함께 Wandb API를 사용하기 위해 초기화하는 데 도움을 줍니다.

기본 사용법은 트레이닝 함수 앞에 `@wandb_mixin` 데코레이터를 추가하는 것입니다:

```python
from ray.tune.integration.wandb import wandb_mixin


@wandb_mixin
def train_fn(config):
    wandb.log()
```

Wandb 설정은 `tune.run()`의 `config` 파라미터에 `wandb 키`를 전달하여 수행됩니다(아래 예제 참조).

wandb config 항목의 내용은 키워드 인수로 `wandb.init()`에 전달됩니다. 다음 설정을 제외하고는, 이 설정들은 `WandbTrainableMixin` 자체를 설정하는데 사용됩니다:

### 파라미터

`api_key_file (str)` – Wandb `API 키`를 포함한 파일의 경로.

`api_key (str)` – Wandb API 키. `api_key_file` 설정의 대안.

Wandb의 `group`, `run_id` 및 `run_name`은 Tune에 의해 자동으로 선택되지만, 해당 설정 값을 채워서 덮어쓸 수 있습니다.

다른 모든 유효한 설정을 보려면 여기를 참조하세요: [https://docs.wandb.com/library/init](https://docs.wandb.com/library/init)

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
        # 검색 공간을 여기에 정의
        "a": tune.choice([1, 2, 3]),
        "b": tune.choice([4, 5, 6]),
        # wandb 설정
        "wandb": {"project": "Optimization_Project", "api_key_file": "/path/to/file"},
    },
)
```

## 예시 코드

통합 작동 방식을 보여주기 위해 몇 가지 예시를 만들었습니다:

* [Colab](http://wandb.me/raytune-colab): 통합을 시도해 볼 수 있는 간단한 데모.
* [대시보드](https://wandb.ai/anmolmann/ray\_tune): 예시에서 생성된 대시보드를 보기.