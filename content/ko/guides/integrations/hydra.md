---
title: Hydra
description: W&B를 Hydra와 연동하는 방법
menu:
  default:
    identifier: ko-guides-integrations-hydra
    parent: integrations
weight: 150
---

> [Hydra](https://hydra.cc)는 연구 및 기타 복잡한 애플리케이션 개발을 단순화하는 오픈소스 Python 프레임워크입니다. Hydra의 핵심 기능은 구성 파일과 커맨드라인을 통해 계층적 설정을 동적으로 생성하고 덮어쓸 수 있는 점입니다.

설정 관리는 계속 Hydra로 하면서도, W&B의 강력함을 함께 활용할 수 있습니다.

## 메트릭 추적하기

`wandb.init()`와 `wandb.Run.log()`로 평소와 같이 메트릭을 추적하세요. 아래 예시에서 `wandb.entity`와 `wandb.project`는 hydra 설정 파일 내에서 정의됩니다.

```python
import wandb


@hydra.main(config_path="configs/", config_name="defaults")
def run_experiment(cfg):

    with wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project) as run:
      run.log({"loss": loss})
```

## 하이퍼파라미터 추적하기

Hydra는 [omegaconf](https://omegaconf.readthedocs.io/en/2.1_branch/)를 기본 설정 사전 인터페이스로 사용합니다. `OmegaConf`의 사전은 일반 사전(dict)의 하위 클래스가 아니기 때문에, Hydra의 `Config` 객체를 바로 `wandb.Run.config`에 전달하면 대시보드에서 의도치 않은 결과가 발생할 수 있습니다. 반드시 `omegaconf.DictConfig`를 기본 `dict` 타입으로 변환한 뒤 `wandb.Run.config`에 전달해야 합니다.

```python
@hydra.main(config_path="configs/", config_name="defaults")
def run_experiment(cfg):
  with wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project) as run:
    run.config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
    run.log({"loss": loss})
    model = Model(**run.config.model.configs)
```

## 멀티프로세싱 문제 해결하기

프로세스 실행 시 멈춘다면, [이미 알려진 이슈]({{< relref path="/guides/models/track/log/distributed-training.md" lang="ko" >}}) 때문일 수 있습니다. 해결 방법으로는 `wandb.init()`에서 extra settings 파라미터를 사용해 멀티프로세싱 프로토콜을 변경하는 방법이 있습니다:

```python
wandb.init(settings=wandb.Settings(start_method="thread"))
```

또는, 셸에서 글로벌 환경 변수를 설정할 수 있습니다:

```bash
$ export WANDB_START_METHOD=thread
```

## 하이퍼파라미터 최적화하기

[W&B Sweeps]({{< relref path="/guides/models/sweeps/" lang="ko" >}})는 매우 확장성 높은 하이퍼파라미터 탐색 플랫폼으로, 적은 코드만으로도 W&B 실험에 대한 다양한 인사이트와 시각화를 제공합니다. Sweeps는 Hydra 프로젝트와 코드 수정 없이 자연스럽게 연동됩니다. 필요한 것은 스윕할 여러 파라미터를 담은 일반 설정 파일 하나뿐입니다.

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

스윕을 실행하세요:

``` bash
wandb sweep sweep.yaml` \
```

W&B는 자동으로 프로젝트 내에 스윕을 만들고, 각 머신에서 스윕을 실행할 수 있도록 `wandb agent` 커맨드를 제공해줍니다.

### Hydra 기본값에 없는 파라미터 전달하기

<a id="pitfall-3-sweep-passing-parameters-not-present-in-defaults"></a>

Hydra는 기본 설정 파일에 없는 추가 파라미터도 커맨드라인에서 `+`를 앞에 붙여 전달하는 것을 지원합니다. 예를 들어, 새로운 파라미터에 값을 넘기고 싶을 땐 아래와 같이 실행할 수 있습니다:

```bash
$ python program.py +experiment=some_experiment
```

이 같은 `+` 설정은 [Hydra Experiments](https://hydra.cc/docs/patterns/configuring_experiments/)를 설정할 때처럼 스윕할 수 없습니다. 이를 해결하려면, 해당 파라미터를 기본 빈 파일로 초기화한 뒤, W&B Sweep을 활용해 각 run마다 빈 설정을 덮어써주면 됩니다. 자세한 내용은 [이 W&B Report](https://wandb.me/hydra)를 참조하세요.