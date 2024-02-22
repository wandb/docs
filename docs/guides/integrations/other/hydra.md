---
description: How to integrate W&B with Hydra.
slug: /guides/integrations/hydra
displayed_sidebar: default
---

# Hydra

> [Hydra](https://hydra.cc)는 연구 및 기타 복잡한 애플리케이션의 개발을 단순화하는 오픈소스 Python 프레임워크입니다. 핵심 기능은 구성 파일 및 명령 줄을 통해 계층적 구성을 동적으로 생성하고 재정의할 수 있는 능력입니다.

W&B의 강력한 기능을 활용하면서 Hydra를 구성 관리에 계속 사용할 수 있습니다.

## 메트릭 추적

`wandb.init`과 `wandb.log`로 평소처럼 메트릭을 추적하세요. 여기에서 `wandb.entity`와 `wandb.project`는 Hydra 구성 파일 내에서 정의됩니다.

```python
import wandb


@hydra.main(config_path="configs/", config_name="defaults")
def run_experiment(cfg):
    run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
    wandb.log({"loss": loss})
```

## 하이퍼파라미터 추적

Hydra는 구성 사전과 상호 작용하는 기본 방법으로 [omegaconf](https://omegaconf.readthedocs.io/en/2.1\_branch/)를 사용합니다. `OmegaConf`의 사전은 기본 사전의 하위 클래스가 아니므로 Hydra의 `Config`를 `wandb.config`에 직접 전달하면 대시보드에서 예상치 못한 결과가 발생합니다. `omegaconf.DictConfig`를 기본 `dict` 타입으로 변환한 후 `wandb.config`에 전달해야 합니다.

```python
@hydra.main(config_path="configs/", config_name="defaults")
def run_experiment(cfg):
    wandb.config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
    wandb.log({"loss": loss})
    model = Model(**wandb.config.model.configs)
```

### 멀티프로세싱 문제 해결

프로세스가 시작될 때 멈춘다면, [이 알려진 문제](../../track/log/distributed-training.md)에 의한 것일 수 있습니다. 이를 해결하기 위해 \`wandb.init\`에 추가 설정 파라미터를 추가하거나:

```
wandb.init(settings=wandb.Settings(start_method="thread"))
```

또는 쉘에서 글로벌 환경 변수를 설정하여 해결해보세요:

```
$ export WANDB_START_METHOD=thread
```

## 하이퍼파라미터 최적화

[W&B Sweeps](../../sweeps/intro.md)는 코드 요구 사항이 최소한으로 W&B 실험에 대한 흥미로운 통찰력과 시각화를 제공하는 고도로 확장 가능한 하이퍼파라미터 검색 플랫폼입니다. Sweeps는 Hydra 프로젝트와 무코딩으로 완벽하게 통합됩니다. 필요한 것은 다양한 파라미터를 스윕할 구성 파일을 정상적으로 설명하는 것뿐입니다.

간단한 예시 `sweep.yaml` 파일은 다음과 같습니다:

```yaml
program: main.py
method: bayes
metric:
  goal: maximize
  name: test/accuracy
parameters:
  dataset:
    values: [mnist, cifar10]

command:
  - ${env}
  - python
  - ${program}
  - ${args_no_hyphens}
```

스윕을 호출하려면:

`wandb sweep sweep.yaml`\
``\
``이를 호출하면 W&B는 프로젝트 내에서 자동으로 스윕을 생성하고 스윕을 실행하려는 각 기계에서 실행할 `wandb agent` 명령을 반환합니다.

#### Hydra 기본값에 없는 파라미터 전달 <a href="#pitfall-3-sweep-passing-parameters-not-present-in-defaults" id="pitfall-3-sweep-passing-parameters-not-present-in-defaults"></a>

Hydra는 기본 구성 파일에 없는 추가 파라미터를 명령 줄을 통해 전달할 수 있도록 지원하며, 명령에 `+`를 사용합니다. 예를 들어, 단순히 다음과 같이 호출하여 추가 파라미터를 어떤 값으로 전달할 수 있습니다:

```
$ python program.py +experiment=some_experiment
```

이러한 `+` 구성을 [Hydra 실험](https://hydra.cc/docs/patterns/configuring\_experiments/)을 구성할 때와 유사하게 스윕할 수는 없습니다. 이를 해결하기 위해, 실험 파라미터를 기본 빈 파일로 초기화하고 각 호출에서 W&B Sweep를 사용하여 이러한 빈 구성을 재정의할 수 있습니다. 자세한 정보는 [**이 W&B 리포트**](http://wandb.me/hydra)**를 참조하세요.**