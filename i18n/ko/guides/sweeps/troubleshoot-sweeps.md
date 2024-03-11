---
description: Troubleshoot common W&B Sweep issues.
displayed_sidebar: default
---

# Sweeps 문제 해결하기

<head>
  <title>W&B Sweeps 문제 해결하기</title>
</head>

안내된 지침을 따라 흔히 발생하는 오류 메시지를 해결하세요.

### `CommError, Run does not exist` 및 `ERROR Error uploading`

이 두 오류 메시지가 모두 반환되면 W&B Run ID가 정의되어 있을 수 있습니다. 예를 들어, Jupyter 노트북이나 Python 스크립트 어딘가에 다음과 유사한 코드조각이 정의되어 있을 수 있습니다:

```python
wandb.init(id="some-string")
```

W&B Sweeps에서 생성된 Run에 대해 W&B가 자동으로 무작위 고유 ID를 생성하기 때문에 W&B Sweeps에 대해 Run ID를 설정할 수 없습니다.

W&B Run ID는 프로젝트 내에서 고유해야 합니다.

W&B를 초기화할 때 name 파라미터에 이름을 전달하면 테이블과 그래프에 표시될 사용자 정의 이름을 설정할 수 있습니다. 예를 들면 다음과 같습니다:

```python
wandb.init(name="a helpful readable run name")
```

### `Cuda out of memory`

이 오류 메시지가 표시되면 코드를 프로세스 기반 실행으로 리팩터링하세요. 보다 구체적으로, 코드를 Python 스크립트로 다시 작성하세요. 또한, W&B Python SDK 대신 CLI에서 W&B 스윕 에이전트를 호출하세요.

예를 들어, 코드를 `train.py`라는 Python 스크립트로 다시 작성한다고 가정합니다. YAML 스윕 구성 파일(`config.yaml`이라고 가정)에 트레이닝 스크립트의 이름(`train.py`)을 추가하세요:

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

CLI로 이동하여 다음과 같이 wandb sweep으로 W&B 스윕을 초기화하세요:

```shell
wandb sweep config.yaml
```

반환된 W&B 스윕 ID를 기록하세요. 다음으로, Python SDK ([`wandb.agent`](../../ref/python/agent.md)) 대신 CLI에서 [`wandb agent`](../../ref/cli/wandb-agent.md)를 사용하여 스윕 작업을 시작하세요. 아래 코드조각에서 `sweep_ID`를 이전 단계에서 반환된 스윕 ID로 교체하세요:

```shell
wandb agent sweep_ID
```

### `anaconda 400 error`

다음 오류는 최적화하고 있는 메트릭을 로그하지 않을 때 일반적으로 발생합니다:

```shell
wandb: ERROR Error while calling W&B API: anaconda 400 error: 
{"code": 400, "message": "TypeError: bad operand type for unary -: 'NoneType'"}
```

YAML 파일이나 중첩된 사전 내에서 최적화할 "metric"이라는 키를 지정합니다. 이 메트릭을 (`wandb.log`) 로그하세요. 또한, Python 스크립트나 Jupyter 노트북 내에서 스윕을 최적화하도록 정의한 _정확한_ 메트릭 이름을 사용하십시오. 설정 파일에 대한 자세한 정보는 [스윕 구성 정의](./define-sweep-configuration.md)를 참조하세요.