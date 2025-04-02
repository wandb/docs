---
title: Sweeps troubleshooting
description: 일반적인 W&B 스윕 문제를 해결합니다.
menu:
  default:
    identifier: ko-guides-models-sweeps-troubleshoot-sweeps
    parent: sweeps
---

다음 가이드라인을 참고하여 자주 발생하는 오류 메시지를 해결하세요.

### `CommError, Run does not exist` 및 `ERROR Error uploading`

이 두 오류 메시지가 모두 반환되는 경우 W&B Run ID가 정의되었을 수 있습니다. 예를 들어 Jupyter Notebook 또는 Python 스크립트에 다음과 유사한 코드 조각이 정의되어 있을 수 있습니다.

```python
wandb.init(id="some-string")
```

W&B는 W&B Sweeps에서 생성된 Runs에 대해 무작위의 고유한 ID를 자동으로 생성하므로 W&B Sweeps에 대한 Run ID를 설정할 수 없습니다.

W&B Run ID는 프로젝트 내에서 고유해야 합니다.

테이블과 그래프에 표시될 사용자 지정 이름을 설정하려면 W&B를 초기화할 때 name 파라미터에 이름을 전달하는 것이 좋습니다. 예:

```python
wandb.init(name="a helpful readable run name")
```

### `Cuda out of memory`

이 오류 메시지가 표시되면 코드 리팩터링을 통해 프로세스 기반 실행을 사용하세요. 특히, 코드를 Python 스크립트로 다시 작성하세요. 또한 W&B Python SDK 대신 CLI에서 W&B Sweep 에이전트를 호출하세요.

예를 들어 코드를 `train.py`라는 Python 스크립트로 다시 작성한다고 가정합니다. 트레이닝 스크립트 이름(`train.py`)을 YAML Sweep 구성 파일(`config.yaml` (이 예시))에 추가합니다.

```yaml
program: train.py
method: bayes
metric:
  name: validation_loss
  goal: maximize
parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
  optimizer:
    values: ["adam", "sgd"]
```

다음으로, 다음 코드를 `train.py` Python 스크립트에 추가합니다.

```python
if _name_ == "_main_":
    train()
```

CLI로 이동하여 wandb sweep 명령어로 W&B Sweep을 초기화합니다.

```shell
wandb sweep config.yaml
```

반환된 W&B Sweep ID를 기록해 둡니다. 다음으로, Python SDK([`wandb.agent`]({{< relref path="/ref/python/agent.md" lang="ko" >}})) 대신 CLI를 사용하여 [`wandb agent`]({{< relref path="/ref/cli/wandb-agent.md" lang="ko" >}})로 Sweep 작업을 시작합니다. 아래 코드 조각에서 `sweep_ID`를 이전 단계에서 반환된 Sweep ID로 바꿉니다.

```shell
wandb agent sweep_ID
```

### `anaconda 400 error`

다음 오류는 일반적으로 최적화하려는 메트릭을 로깅하지 않을 때 발생합니다.

```shell
wandb: ERROR Error while calling W&B API: anaconda 400 error: 
{"code": 400, "message": "TypeError: bad operand type for unary -: 'NoneType'"}
```

YAML 파일 또는 중첩된 사전 내에서 최적화할 "metric"이라는 키를 지정합니다. 이 메트릭을 반드시 로깅(`wandb.log`)해야 합니다. 또한 Python 스크립트 또는 Jupyter Notebook 내에서 스윕을 최적화하도록 정의한 _정확한_ 메트릭 이름을 사용해야 합니다. 구성 파일에 대한 자세한 내용은 [스윕 구성 정의]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ko" >}})를 참조하세요.
