---
title: Hydra
description: Hydra와 W&B를 통합하는 방법
menu:
  default:
    identifier: ko-guides-integrations-hydra
    parent: integrations
weight: 150
---

> [Hydra](https://hydra.cc)는 연구 및 기타 복잡한 애플리케이션 개발을 간소화하는 오픈 소스 Python 프레임워크입니다. 주요 기능은 구성 파일을 통해 계층적 설정을 동적으로 생성하고 커맨드라인을 통해 재정의할 수 있는 기능입니다.

W&B의 강력한 기능을 활용하면서 Hydra를 계속 사용하여 설정을 관리할 수 있습니다.

## 메트릭 추적

`wandb.init` 및 `wandb.log`를 사용하여 평소와 같이 메트릭을 추적합니다. 여기서 `wandb.entity` 및 `wandb.project`는 hydra 설정 파일 내에 정의됩니다.

```python
import wandb


@hydra.main(config_path="configs/", config_name="defaults")
def run_experiment(cfg):
    run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
    wandb.log({"loss": loss})
```

## 하이퍼파라미터 추적

Hydra는 `omegaconf` ([https://omegaconf.readthedocs.io/en/2.1_branch/](https://omegaconf.readthedocs.io/en/2.1_branch/))를 설정 dictionary와 상호 작용하는 기본 방식으로 사용합니다. `OmegaConf`의 dictionary는 기본 dictionary의 하위 클래스가 아니므로 Hydra의 `Config`를 `wandb.config`에 직접 전달하면 대시보드에서 예기치 않은 결과가 발생합니다. `omegaconf.DictConfig`를 기본 `dict` 유형으로 변환한 다음 `wandb.config`에 전달해야 합니다.

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

## 멀티프로세싱 문제 해결

프로세스가 시작될 때 멈추는 경우 [알려진 문제]({{< relref path="/guides/models/track/log/distributed-training.md" lang="ko" >}})로 인해 발생할 수 있습니다. 이 문제를 해결하려면 다음과 같이 추가 설정 파라미터를 \`wandb.init\`에 추가하여 wandb의 멀티프로세싱 프로토콜을 변경해 보십시오.

```python
wandb.init(settings=wandb.Settings(start_method="thread"))
```

또는 셸에서 전역 환경 변수를 설정합니다.

```bash
$ export WANDB_START_METHOD=thread
```

## 하이퍼파라미터 최적화

[W&B Sweeps]({{< relref path="/guides/models/sweeps/" lang="ko" >}})는 확장성이 뛰어난 하이퍼파라미터 검색 플랫폼으로, 최소한의 코딩으로 W&B Experiments에 대한 흥미로운 통찰력과 시각화를 제공합니다. Sweeps는 코딩 없이도 Hydra Projects와 원활하게 통합됩니다. 스윕할 다양한 파라미터를 설명하는 설정 파일만 있으면 됩니다.

간단한 `sweep.yaml` 파일 예제는 다음과 같습니다.

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

스윕을 호출합니다.

``` bash
wandb sweep sweep.yaml` \
```

W&B는 자동으로 프로젝트 내부에 스윕을 생성하고 스윕을 실행할 각 머신에서 실행할 `wandb agent` 코맨드를 반환합니다.

### Hydra 기본값에 없는 파라미터 전달

<a id="pitfall-3-sweep-passing-parameters-not-present-in-defaults"></a>

Hydra는 커맨드 앞에 `+`를 사용하여 기본 설정 파일에 없는 추가 파라미터를 커맨드라인을 통해 전달하는 것을 지원합니다. 예를 들어 다음과 같이 호출하여 일부 값이 있는 추가 파라미터를 전달할 수 있습니다.

```bash
$ python program.py +experiment=some_experiment
```

[Hydra Experiments](https://hydra.cc/docs/patterns/configuring_experiments/)를 구성하는 동안 수행하는 작업과 유사하게 이러한 `+` 설정에 대해 스윕할 수 없습니다. 이를 해결하려면 기본 빈 파일로 experiment 파라미터를 초기화하고 W&B Sweep을 사용하여 각 호출에서 해당 빈 설정을 재정의할 수 있습니다. 자세한 내용은 [**이 W&B Report**](http://wandb.me/hydra)**를 참조하십시오.**
