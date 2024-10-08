---
title: Hydra
description: W&B를 Hydra와 통합하는 방법.
slug: /guides/integrations/hydra
displayed_sidebar: default
---

> [Hydra](https://hydra.cc)는 연구 및 기타 복잡한 애플리케이션 개발을 단순화하는 오픈 소스 Python 프레임워크입니다. 핵심 기능은 구성 파일 및 커맨드라인을 통해 구성 및 재정의할 수 있는 계층적 설정을 동적 생성할 수 있는 능력입니다.

W&B의 강력한 기능을 활용하면서 설정 관리에 Hydra를 계속 사용할 수 있습니다.

## 메트릭 추적

`wandb.init` 및 `wandb.log`로 메트릭을 평소와 같이 추적하세요. 여기서 `wandb.entity`와 `wandb.project`는 hydra 설정 파일 내에서 정의됩니다.

```python
import wandb


@hydra.main(config_path="configs/", config_name="defaults")
def run_experiment(cfg):
    run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
    wandb.log({"loss": loss})
```

## 하이퍼파라미터 추적

Hydra는 설정 사전 인터페이스의 기본 방법으로 [omegaconf](https://omegaconf.readthedocs.io/en/2.1_branch/)를 사용합니다. `OmegaConf`의 사전은 원시 사전의 하위 클래스가 아니기 때문에 Hydra의 `Config`를 `wandb.config`로 직접 전달하면 대시보드에서 예기치 않은 결과가 발생할 수 있습니다. `omegaconf.DictConfig`를 원시 `dict` 유형으로 변환한 후 `wandb.config`로 전달해야 합니다.

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

프로세스가 시작할 때 정지하는 경우, 이는 [이 알려진 문제](../../track/log/distributed-training.md)로 인해 발생할 수 있습니다. 이를 해결하려면 `wandb.init`에 추가 설정 파라미터를 추가하여 wandb의 멀티프로세싱 프로토콜을 변경해 보세요:

```
wandb.init(settings=wandb.Settings(start_method="thread"))
```

또는 셸에서 전역 환경 변수를 설정하세요:

```
$ export WANDB_START_METHOD=thread
```

## 하이퍼파라미터 최적화

[W&B Sweeps](../../sweeps/intro.md)은 최소한의 코드 요구로 W&B 실험에 대한 흥미로운 통찰력과 시각화를 제공하는 매우 확장 가능한 하이퍼파라미터 검색 플랫폼입니다. Sweeps는 추가 코딩 요구 없이 Hydra 프로젝트와 원활하게 통합됩니다. 필요한 것은 다양한 파라미터를 설명하는 설정 파일뿐입니다.

간단한 예 `sweep.yaml` 파일은 다음과 같습니다:

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

스윕을 호출하려면 다음을 사용하세요:

`wandb sweep sweep.yaml`

이 코맨드를 호출하면, W&B가 자동으로 프로젝트 내에 스윕을 생성하고 실행할 각 머신에 적용할 `wandb agent` 코맨드를 반환합니다.

#### Hydra 기본값에 없는 파라미터 전달 <a href="#pitfall-3-sweep-passing-parameters-not-present-in-defaults" id="pitfall-3-sweep-passing-parameters-not-present-in-defaults"></a>

Hydra는 커맨드라인을 통해 기본 설정 파일에 없는 추가 파라미터를 `+`를 사용하여 전달할 수 있도록 지원합니다. 예를 들어, 어떤 값을 가진 추가 파라미터를 전달하려면 다음과 같이 호출할 수 있습니다:

```
$ python program.py +experiment=some_experiment
```

이러한 `+` 설정을 Hydra Experiments을 구성할 때처럼 스윕할 수 없습니다. 이를 해결하려면 빈 파일을 기본값으로 실험 파라미터를 초기화하고 각 호출 시 W&B Sweep을 사용하여 빈 설정을 재정의할 수 있습니다. 자세한 내용은 [**이 W&B 리포트**](http://wandb.me/hydra)를 참조하세요.