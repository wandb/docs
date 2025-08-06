---
title: Sweeps 문제 해결
description: 일반적인 W&B 스윕 문제 해결 방법.
menu:
  default:
    identifier: ko-guides-models-sweeps-troubleshoot-sweeps
    parent: sweeps
---

안내에 따라 자주 발생하는 오류 메시지들을 해결하는 방법을 안내합니다.

### `CommError, Run does not exist` 및 `ERROR Error uploading`

이 두 가지 오류 메시지가 동시에 나타날 경우, W&B Run ID가 명시적으로 정의되어 있을 수 있습니다. 예를 들어, Jupyter Notebook이나 Python 스크립트 어딘가에 다음과 같은 코드조각이 있을 수 있습니다:

```python
wandb.init(id="some-string")
```

W&B Sweeps에서는 Run ID를 직접 지정할 수 없습니다. Sweeps에서 생성되는 Run들은 W&B가 자동으로 무작위의 고유 ID를 생성하기 때문입니다.

W&B Run ID는 프로젝트 내에서 고유해야 합니다.

만약 테이블이나 그래프에 나타날 사용자 지정 이름을 설정하고 싶다면, W&B 초기화 시 name 파라미터에 이름을 넘기는 방법을 권장합니다. 예:

```python
wandb.init(name="a helpful readable run name")
```

### `Cuda out of memory`

이 오류 메시지가 나타난다면, 코드가 프로세스 기반 실행 방식을 사용하도록 리팩토링하세요. 즉, 코드를 Python 스크립트로 작성하고, W&B Python SDK 대신 CLI에서 W&B Sweep Agent를 호출하는 것이 좋습니다.

예를 들어, 코드를 `train.py`라는 Python 스크립트로 변경했다고 가정합니다. 그런 후, 트레이닝 스크립트 이름(`train.py`)을 YAML Sweep 설정 파일(`config.yaml` 예시)에서 다음과 같이 지정합니다:

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

다음으로, `train.py` Python 스크립트에 아래 코드를 추가하세요:

```python
if _name_ == "_main_":
    train()
```

CLI로 이동하여 wandb sweep 명령어로 W&B Sweep을 초기화하세요:

```shell
wandb sweep config.yaml
```

반환되는 W&B Sweep ID를 메모하세요. 다음 단계에서는 Python SDK([`wandb.agent`]({{< relref path="/ref/python/sdk/functions/agent.md" lang="ko" >}})) 대신 CLI에서 [`wandb agent`]({{< relref path="/ref/cli/wandb-agent.md" lang="ko" >}})로 Sweep 작업을 실행합니다. 아래 코드조각의 `sweep_ID`는 앞서 받은 Sweep ID로 바꿔주세요:

```shell
wandb agent sweep_ID
```

### `anaconda 400 error`

다음 오류는 최적화하려는 metric을 로그하지 않았을 때 주로 발생합니다:

```shell
wandb: ERROR Error while calling W&B API: anaconda 400 error: 
{"code": 400, "message": "TypeError: bad operand type for unary -: 'NoneType'"}
```

YAML 파일 또는 중첩 사전에서 "metric"이라는 키로 최적화할 지표를 지정합니다. 반드시 해당 metric을 `wandb.log`로 로그해야 합니다. 그리고, Python 스크립트나 Jupyter Notebook에서도 스윕에서 정의한 *정확한* metric 이름을 사용해야 합니다. 설정 파일에 대한 자세한 정보는 [스윕 구성 정의]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ko" >}})를 참고하세요.