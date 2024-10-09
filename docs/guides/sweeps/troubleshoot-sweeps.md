---
title: Sweeps troubleshooting
description: 일반적인 W&B 스윕 문제 해결.
displayed_sidebar: default
---

일반적인 오류 메시지를 해결하기 위한 제안된 가이드를 참조하세요.

### `CommError, Run does not exist` 및 `ERROR Error uploading`

두 가지 오류 메시지가 모두 반환된다면 W&B Run ID가 정의되어 있을 수 있습니다. 예를 들어, 당신이 Jupyter Notebook이나 Python 스크립트 어딘가에 유사한 코드조각을 정의했을 수 있습니다:

```python
wandb.init(id="some-string")
```

W&B Sweeps에 대해 Run ID를 설정할 수 없습니다. 왜냐하면 W&B는 Sweeps에 의해 생성된 Runs에 대해 무작위로 고유한 ID를 자동으로 생성하기 때문입니다.

W&B Run ID는 프로젝트 내에서 고유해야 합니다.

사용자 정의 이름을 테이블 및 그래프에 나타내고 싶다면, W&B를 초기화할 때 name 파라미터에 이름을 전달하는 것을 추천합니다. 예를 들어:

```python
wandb.init(name="a helpful readable run name")
```

### `Cuda out of memory`

코드를 리팩터링하여 프로세스 기반 실행을 사용하도록 수정하세요, 만약 이 오류 메시지를 본다면. 보다 구체적으로, 코드를 Python 스크립트로 다시 작성하세요. 추가로, W&B Python SDK 대신 CLI에서 W&B Sweep Agent를 호출하세요.

예를 들어, 코드가 `train.py`라는 Python 스크립트로 다시 작성되었다고 가정합니다. 트레이닝 스크립트의 이름(`train.py`)을 YAML 스윕 구성 파일(`config.yaml` 이 예제에서)에 추가하세요:

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

다음으로, `train.py` Python 스크립트에 다음을 추가하세요:

```python
if _name_ == "_main_":
    train()
```

CLI로 이동하여 wandb sweep으로 W&B Sweep을 초기화하세요:

```shell
wandb sweep config.yaml
```

반환된 W&B Sweep ID를 기록하십시오. 다음으로, Python SDK([`wandb.agent`](../../ref/python/agent.md)) 대신 CLI에서 [`wandb agent`](../../ref/cli/wandb-agent.md)를 사용하여 Sweep 작업을 시작하십시오. 이전 단계에서 반환된 Sweep ID로 아래 코드조각의 `sweep_ID`를 교체하십시오:

```shell
wandb agent sweep_ID
```

### `anaconda 400 error`

다음 오류는 최적화하고 있는 메트릭을 기록하지 않을 때 일반적으로 발생합니다:

```shell
wandb: ERROR Error while calling W&B API: anaconda 400 error: 
{"code": 400, "message": "TypeError: bad operand type for unary -: 'NoneType'"}
```

YAML 파일이나 중첩된 사전에서 최적화할 "metric"이라는 이름의 키를 지정합니다. 이 메트릭을 (`wandb.log`) 기록하는지 확인하십시오. 추가로, Python 스크립트나 Jupyter Notebook 내에서 스윕을 최적화하도록 정의한 *정확한* 메트릭 이름을 사용하는지 확인하십시오. 구성 파일에 대한 자세한 정보는 [Define sweep configuration](./define-sweep-configuration.md)을 참조하십시오.