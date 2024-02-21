---
description: How to integrate W&B with Ray Tune.
slug: /guides/integrations/ray-tune
displayed_sidebar: default
---

# Ray Tune

W&B는 두 가지 가벼운 통합을 제공하여 [Ray](https://github.com/ray-project/ray)와 통합됩니다.

하나는 `WandbLoggerCallback`으로, Tune에 보고된 메트릭을 Wandb API로 자동 로깅합니다. 다른 하나는 함수 API와 함께 사용할 수 있는 `@wandb_mixin` 데코레이터로, Wandb API를 Tune의 학습 정보와 함께 자동으로 초기화합니다. 일반적으로 할 때처럼 Wandb API를 사용할 수 있습니다. 예를 들어, 학습 프로세스를 로깅하기 위해 `wandb.log()`를 사용합니다.

## WandbLoggerCallback

```python
from ray.air.integrations.wandb import WandbLoggerCallback
```

Wandb 구성은 `tune.run()`의 config 파라미터에 wandb 키를 전달하여 수행됩니다(아래 예제 참조).

wandb config 항목의 내용은 키워드 인수로 `wandb.init()`에 전달됩니다. 다음 설정을 제외하고는 `WandbLoggerCallback` 자체를 구성하는 데 사용됩니다:

### 파라미터

`api_key_file (str)` – `Wandb API 키`를 포함한 파일의 경로입니다.

`api_key (str)` – Wandb API 키입니다. `api_key_file`을 설정하는 대안입니다.

`excludes (list)` – 로그에서 제외해야 하는 메트릭의 목록입니다.

`log_config (bool)` – 결과 dict의 config 파라미터가 로깅되어야 하는지 여부를 나타내는 부울입니다. 예를 들어, `PopulationBasedTraining`과 같이 학습 중에 파라미터가 변경될 경우 이는 유용합니다. 기본값은 False입니다.

### 예제

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

이 Ray Tune Trainable `mixin`은 `Trainable` 클래스 또는 함수 API와 함께 `@wandb_mixin`을 사용하기 위해 Wandb API를 초기화하는 데 도움이 됩니다.

기본 사용법은 학습 함수 앞에 `@wandb_mixin` 데코레이터를 추가하는 것입니다:

```python
from ray.tune.integration.wandb import wandb_mixin


@wandb_mixin
def train_fn(config):
    wandb.log()
```

Wandb 구성은 `tune.run()`의 `config` 파라미터에 `wandb 키`를 전달하여 수행됩니다(아래 예제 참조).

wandb config 항목의 내용은 키워드 인수로 `wandb.init()`에 전달됩니다. 다음 설정을 제외하고는 `WandbTrainableMixin` 자체를 구성하는 데 사용됩니다:

### 파라미터

`api_key_file (str)` – Wandb `API 키`를 포함한 파일의 경로입니다.

`api_key (str)` – Wandb API 키입니다. `api_key_file`을 설정하는 대안입니다.

Wandb의 `group`, `run_id` 및 `run_name`은 Tune에 의해 자동으로 선택되지만, 해당 구성 값을 입력하여 덮어쓸 수 있습니다.

다른 모든 유효한 구성 설정은 여기서 확인하십시오: [https://docs.wandb.com/library/init](https://docs.wandb.com/library/init)

### 예제:

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
        # 여기에 검색 공간 정의
        "a": tune.choice([1, 2, 3]),
        "b": tune.choice([4, 5, 6]),
        # wandb 구성
        "wandb": {"project": "Optimization_Project", "api_key_file": "/path/to/file"},
    },
)
```

## 예제 코드

통합이 어떻게 작동하는지 확인할 수 있는 몇 가지 예제를 만들었습니다:

* [Colab](http://wandb.me/raytune-colab): 통합을 시험해 볼 수 있는 간단한 데모입니다.
* [대시보드](https://wandb.ai/anmolmann/ray_tune): 예제에서 생성된 대시보드를 보십시오.