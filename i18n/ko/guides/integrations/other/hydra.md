---
description: How to integrate W&B with Hydra.
slug: /guides/integrations/hydra
displayed_sidebar: default
---

# Hydra

> [Hydra](https://hydra.cc)는 연구 및 기타 복잡한 애플리케이션의 개발을 단순화하는 오픈 소스 Python 프레임워크입니다. 주요 기능은 구성을 통한 계층적 설정을 동적으로 생성하고 설정 파일 및 커맨드라인을 통해 이를 재정의할 수 있는 능력입니다.

W&B의 강력함을 활용하면서도 Hydra를 설정 관리에 계속 사용할 수 있습니다.

## 메트릭 트래킹

`wandb.init` 및 `wandb.log`를 사용하여 평소와 같이 메트릭을 트래킹합니다. 여기서 `wandb.entity`와 `wandb.project`는 hydra 설정 파일 내에 정의됩니다.

```python
import wandb


@hydra.main(config_path="configs/", config_name="defaults")
def run_experiment(cfg):
    run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
    wandb.log({"loss": loss})
```

## 하이퍼파라미터 트래킹

Hydra는 설정 사전과 인터페이스하는 기본 방법으로 [omegaconf](https://omegaconf.readthedocs.io/en/2.1\_branch/)를 사용합니다. `OmegaConf`의 사전은 원시 사전의 하위 클래스가 아니므로 Hydra의 `Config`를 `wandb.config`에 직접 전달하면 대시보드에서 예상치 못한 결과가 발생합니다. `omegaconf.DictConfig`를 원시 `dict` 유형으로 변환한 후에 `wandb.config`로 전달해야 합니다.

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

프로세스가 시작될 때 멈추는 경우, [이 알려진 문제](../../track/log/distributed-training.md)에 의해 발생할 수 있습니다. 이를 해결하기 위해, 다음과 같이 `wandb.init`에 추가 설정 파라미터를 추가하여 wandb의 멀티프로세싱 프로토콜을 변경하거나:

```
wandb.init(settings=wandb.Settings(start_method="thread"))
```

또는 셸에서 전역 환경 변수를 설정하여:

```
$ export WANDB_START_METHOD=thread
```

## 하이퍼파라미터 최적화

[W&B Sweeps](../../sweeps/intro.md)는 최소한의 코드 요구사항으로 흥미로운 인사이트와 시각화를 제공하는 고도로 확장 가능한 하이퍼파라미터 검색 플랫폼입니다. Sweeps는 Hydra 프로젝트와 무코딩 요구사항으로 완벽하게 통합됩니다. 필요한 것은 스윕할 다양한 파라미터를 설명하는 구성 파일뿐입니다.

간단한 예제 `sweep.yaml` 파일은 다음과 같습니다:

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
``이를 호출하면 W&B는 프로젝트 내에 자동으로 스윕을 생성하고 스윕을 실행하고 싶은 각 기계에서 실행할 `wandb 에이전트` 코맨드를 반환합니다.

#### Hydra 디폴트에 없는 파라미터 전달 <a href="#pitfall-3-sweep-passing-parameters-not-present-in-defaults" id="pitfall-3-sweep-passing-parameters-not-present-in-defaults"></a>

Hydra는 기본 구성 파일에 없는 추가 파라미터를 커맨드라인을 통해 전달할 수 있는 기능을 지원합니다. 이는 커맨드 앞에 `+`를 사용함으로써 가능합니다. 예를 들어, 추가 파라미터와 그 값을 호출하려면 다음과 같이 하면 됩니다:

```
$ python program.py +experiment=some_experiment
```

이러한 `+` 구성을 [Hydra 실험](https://hydra.cc/docs/patterns/configuring\_experiments/)을 구성할 때와 같이 스윕할 수는 없습니다. 이를 해결하기 위해, 실험 파라미터를 기본 빈 파일로 초기화하고 각 호출에서 W&B 스윕을 사용해 이러한 빈 구성을 재정의할 수 있습니다. 자세한 정보는 [**이 W&B 리포트**](http://wandb.me/hydra)**를 참조하십시오.**