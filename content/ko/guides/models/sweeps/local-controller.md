---
title: Manage algorithms locally
description: W&B 클라우드 호스팅 서비스를 사용하는 대신 로컬에서 알고리즘을 검색하고 중지합니다.
menu:
  default:
    identifier: ko-guides-models-sweeps-local-controller
    parent: sweeps
---

하이퍼파라미터 컨트롤러는 기본적으로 Weights & Biased에서 클라우드 서비스로 호스팅됩니다. W&B 에이전트는 컨트롤러와 통신하여 트레이닝에 사용할 다음 파라미터 세트를 결정합니다. 컨트롤러는 또한 중단할 run을 결정하기 위해 조기 중단 알고리즘을 실행하는 역할도 합니다.

로컬 컨트롤러 기능을 통해 사용자는 검색을 시작하고 알고리즘을 로컬에서 중단할 수 있습니다. 로컬 컨트롤러는 사용자에게 문제를 디버깅하고 클라우드 서비스에 통합할 수 있는 새로운 기능을 개발하기 위해 코드를 검사하고 계측할 수 있는 기능을 제공합니다.

{{% alert color="secondary" %}}
이 기능은 Sweeps 툴에 대한 새로운 알고리즘의 더 빠른 개발 및 디버깅을 지원하기 위해 제공됩니다. 실제 하이퍼파라미터 최적화 워크로드에는 적합하지 않습니다.
{{% /alert %}}

시작하기 전에 W&B SDK(`wandb`)를 설치해야 합니다. 커맨드 라인에 다음 코드 조각을 입력하세요.

```
pip install wandb sweeps
```

다음 예제에서는 이미 구성 파일과 트레이닝 루프가 Python 스크립트 또는 Jupyter Notebook에 정의되어 있다고 가정합니다. 구성 파일을 정의하는 방법에 대한 자세한 내용은 [스윕 구성 정의]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ko" >}})을 참조하세요.

### 커맨드 라인에서 로컬 컨트롤러 실행

W&B에서 클라우드 서비스로 호스팅하는 하이퍼파라미터 컨트롤러를 사용할 때와 유사하게 스윕을 초기화합니다. 컨트롤러 플래그(`controller`)를 지정하여 W&B 스윕 작업에 로컬 컨트롤러를 사용하려는 것을 나타냅니다.

```bash
wandb sweep --controller config.yaml
```

또는 스윕 초기화와 로컬 컨트롤러 사용 지정 단계를 분리할 수 있습니다.

단계를 분리하려면 먼저 다음 키-값을 스윕의 YAML 구성 파일에 추가합니다.

```yaml
controller:
  type: local
```

다음으로 스윕을 초기화합니다.

```bash
wandb sweep config.yaml
```

스윕을 초기화한 후 [`wandb controller`]({{< relref path="/ref/python/controller.md" lang="ko" >}})로 컨트롤러를 시작합니다.

```bash
# wandb sweep 코맨드는 sweep_id를 출력합니다.
wandb controller {user}/{entity}/{sweep_id}
```

로컬 컨트롤러를 사용하겠다고 지정했으면 스윕을 실행하기 위해 하나 이상의 Sweep 에이전트를 시작합니다. 일반적으로 하는 것처럼 W&B 스윕을 시작합니다. 자세한 내용은 [스윕 에이전트 시작]({{< relref path="/guides/models/sweeps/start-sweep-agents.md" lang="ko" >}})을 참조하세요.

```bash
wandb sweep sweep_ID
```

### W&B Python SDK로 로컬 컨트롤러 실행

다음 코드 조각은 W&B Python SDK로 로컬 컨트롤러를 지정하고 사용하는 방법을 보여줍니다.

Python SDK로 컨트롤러를 사용하는 가장 간단한 방법은 스윕 ID를 [`wandb.controller`]({{< relref path="/ref/python/controller.md" lang="ko" >}}) 메서드에 전달하는 것입니다. 다음으로 반환 오브젝트 `run` 메서드를 사용하여 스윕 작업을 시작합니다.

```python
sweep = wandb.controller(sweep_id)
sweep.run()
```

컨트롤러 루프를 더 세밀하게 제어하려면 다음을 수행하세요.

```python
import wandb

sweep = wandb.controller(sweep_id)
while not sweep.done():
    sweep.print_status()
    sweep.step()
    time.sleep(5)
```

또는 제공되는 파라미터를 훨씬 더 제어할 수 있습니다.

```python
import wandb

sweep = wandb.controller(sweep_id)
while not sweep.done():
    params = sweep.search()
    sweep.schedule(params)
    sweep.print_status()
```

코드로 스윕을 완전히 지정하려면 다음과 같이 할 수 있습니다.

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
