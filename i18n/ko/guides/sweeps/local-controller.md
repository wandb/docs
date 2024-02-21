---
description: Search and stop algorithms locally instead of using the W&B cloud-hosted
  service.
displayed_sidebar: default
---

# 로컬에서 검색 및 중지 알고리즘 실행

<head>
  <title>W&B 에이전트로 로컬에서 검색 및 중지 알고리즘 실행</title>
</head>

하이퍼파라미터 컨트롤러는 기본적으로 Weights & Biases가 클라우드 서비스로 호스팅합니다. W&B 에이전트는 컨트롤러와 통신하여 학습에 사용할 다음 세트의 파라미터를 결정합니다. 컨트롤러는 또한 어떤 실행을 중지할 수 있는지 결정하기 위해 조기 중지 알고리즘을 실행하는 책임도 가집니다.

로컬 컨트롤러 기능은 사용자가 로컬에서 검색 및 중지 알고리즘을 시작할 수 있게 해줍니다. 로컬 컨트롤러는 사용자가 코드를 검사하고 조작하여 문제를 디버그하고 클라우드 서비스에 통합될 수 있는 새로운 기능을 개발할 수 있는 능력을 제공합니다.

:::caution
이 기능은 스윕 도구에 대한 새로운 알고리즘 개발 및 디버깅을 더 빠르게 지원하기 위해 제공됩니다. 실제 하이퍼파라미터 최적화 작업에는 의도된 것이 아닙니다.
:::

시작하기 전에, W&B SDK(`wandb`)를 설치해야 합니다. 명령 줄에 다음 코드 조각을 입력하세요:

```
pip install wandb sweeps 
```

다음 예제는 구성 파일과 파이썬 스크립트나 Jupyter 노트북에 정의된 학습 루프가 이미 있다고 가정합니다. 구성 파일을 정의하는 방법에 대한 자세한 정보는 [스윕 구성 정의](./define-sweep-configuration.md)를 참조하세요.

### 명령 줄에서 로컬 컨트롤러 실행

W&B가 클라우드 서비스로 호스팅하는 하이퍼파라미터 컨트롤러를 사용할 때와 유사하게 스윕을 초기화하세요. 로컬 컨트롤러를 사용하고 싶다는 것을 나타내기 위해 컨트롤러 플래그(`controller`)를 지정하세요:

```bash
wandb sweep --controller config.yaml
```

또는, 스윕을 초기화하고 로컬 컨트롤러를 사용하고 싶다는 것을 두 단계로 나눌 수 있습니다.

단계를 분리하려면, 먼저 스윕의 YAML 구성 파일에 다음 키-값을 추가하세요:

```yaml
controller:
  type: local
```

다음으로, 스윕을 초기화하세요:

```bash
wandb sweep config.yaml
```

스윕을 초기화한 후, [`wandb controller`](../../ref/python/controller.md)로 컨트롤러를 시작하세요:

```bash
# wandb sweep 명령은 sweep_id를 출력할 것입니다
wandb controller {user}/{entity}/{sweep_id}
```

로컬 컨트롤러를 사용하겠다고 지정한 후, 스윕을 실행할 하나 이상의 스윕 에이전트를 시작하세요. 보통 할 때처럼 W&B 스윕을 시작하세요. 더 많은 정보는 [스윕 에이전트 시작](../../guides/sweeps/start-sweep-agents.md)을 참조하세요.

```bash
wandb sweep sweep_ID
```

### W&B Python SDK로 로컬 컨트롤러 실행

다음 코드 조각은 W&B Python SDK로 로컬 컨트롤러를 지정하고 사용하는 방법을 보여줍니다.

Python SDK로 컨트롤러를 사용하는 가장 간단한 방법은 스윕 ID를 [`wandb.controller`](../../ref/python/controller.md) 메서드에 전달하는 것입니다. 다음으로, 반환된 객체의 `run` 메서드를 사용하여 스윕 작업을 시작하세요:

```python
sweep = wandb.controller(sweep_id)
sweep.run()
```

컨트롤러 루프를 더 컨트롤하고 싶다면:

```python
import wandb

sweep = wandb.controller(sweep_id)
while not sweep.done():
    sweep.print_status()
    sweep.step()
    time.sleep(5)
```

서빙되는 파라미터에 대해 더 많은 컨트롤을 원한다면:

```python
import wandb

sweep = wandb.controller(sweep_id)
while not sweep.done():
    params = sweep.search()
    sweep.schedule(params)
    sweep.print_status()
```

코드로 전체 스윕을 지정하고 싶다면 다음과 같이 할 수 있습니다:

```python
import wandb

sweep = wandb.controller()
sweep.configure_search("grid")
sweep.configure_program("train-dummy.py")
sweep.configure_controller(type="local")
sweep.configure_parameter("param1", value=3)
sweep.create()
sweep.run()
```