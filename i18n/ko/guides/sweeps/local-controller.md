---
description: Search and stop algorithms locally instead of using the W&B cloud-hosted
  service.
displayed_sidebar: default
---

# 로컬에서 검색 및 중지 알고리즘

<head>
  <title>W&B 에이전트를 사용하여 로컬에서 검색 및 중지 알고리즘 실행</title>
</head>

하이퍼파라미터 컨트롤러는 기본적으로 Weights & Biased에 의해 클라우드 서비스로 호스팅됩니다. W&B 에이전트는 트레이닝을 위해 사용할 다음 세트의 파라미터를 결정하기 위해 컨트롤러와 통신합니다. 컨트롤러는 또한 어떤 run을 중지할 수 있는지 결정하기 위해 조기 중지 알고리즘을 실행하는 책임도 있습니다.

로컬 컨트롤러 기능은 사용자가 로컬에서 검색 및 중지 알고리즘을 시작할 수 있게 합니다. 로컬 컨트롤러는 사용자가 코드를 검사하고 문제를 디버깅하거나 클라우드 서비스에 통합될 수 있는 새로운 기능을 개발할 수 있는 능력을 제공합니다.

:::caution
이 기능은 Sweeps 툴에 대한 새로운 알고리즘의 개발 및 디버깅을 더 빠르게 지원하기 위해 제공됩니다. 실제 하이퍼파라미터 최적화 작업을 위한 것이 아닙니다.
:::

시작하기 전에, W&B SDK(`wandb`)를 설치해야 합니다. 다음 코드조각을 커맨드라인에 입력하세요:

```
pip install wandb sweeps 
```

다음 예제는 이미 설정 파일과 파이썬 스크립트나 Jupyter 노트북에 정의된 트레이닝 루프를 가지고 있다고 가정합니다. 설정 파일을 정의하는 방법에 대한 자세한 내용은 [스윕 구성 정의하기](./define-sweep-configuration.md)를 참조하세요.

### 커맨드라인에서 로컬 컨트롤러 실행하기

W&B 클라우드 서비스에서 호스팅된 하이퍼파라미터 컨트롤러를 사용할 때와 비슷하게 스윕을 초기화합니다. 로컬 컨트롤러를 사용하고 싶다는 것을 나타내기 위해 컨트롤러 플래그(`controller`)를 지정하세요:

```bash
wandb sweep --controller config.yaml
```

또는, 스윕을 초기화하는 것과 로컬 컨트롤러를 사용하고 싶다는 것을 나타내는 것을 두 단계로 분리할 수 있습니다.

단계를 분리하려면, 먼저 스윕의 YAML 설정 파일에 다음 키-값을 추가하세요:

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
# wandb sweep 코맨드는 sweep_id를 출력할 것입니다
wandb controller {user}/{entity}/{sweep_id}
```

로컬 컨트롤러를 사용하겠다고 지정한 후에는, 하나 이상의 스윕 에이전트를 시작하여 스윕을 실행합니다. 보통처럼 W&B 스윕을 시작합니다. 더 자세한 정보는 [스윕 에이전트 시작하기](../../guides/sweeps/start-sweep-agents.md)를 참고하세요.

```bash
wandb sweep sweep_ID
```

### W&B Python SDK로 로컬 컨트롤러 실행하기

다음 코드조각은 W&B Python SDK로 로컬 컨트롤러를 지정하고 사용하는 방법을 보여줍니다.

Python SDK와 컨트롤러를 사용하는 가장 간단한 방법은 [`wandb.controller`](../../ref/python/controller.md) 메소드에 스윕 ID를 전달하는 것입니다. 다음으로, 반환된 오브젝트의 `run` 메소드를 사용하여 스윕 작업을 시작하세요:

```python
sweep = wandb.controller(sweep_id)
sweep.run()
```

컨트롤러 루프를 더 많이 제어하고 싶다면:

```python
import wandb

sweep = wandb.controller(sweep_id)
while not sweep.done():
    sweep.print_status()
    sweep.step()
    time.sleep(5)
```

제공되는 파라미터를 더 많이 제어하려면:

```python
import wandb

sweep = wandb.controller(sweep_id)
while not sweep.done():
    params = sweep.search()
    sweep.schedule(params)
    sweep.print_status()
```

코드로만 스윕을 지정하고 싶다면 다음과 같이 할 수 있습니다:

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