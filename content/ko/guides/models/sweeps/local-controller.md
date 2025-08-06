---
title: 알고리즘을 로컬에서 관리하기
description: W&B 클라우드에 호스팅된 서비스 대신 로컬에서 Search 및 stop 알고리즘을 실행하세요.
menu:
  default:
    identifier: ko-guides-models-sweeps-local-controller
    parent: sweeps
---

하이퍼파라미터 컨트롤러는 기본적으로 Weights & Biases에서 클라우드 서비스로 호스팅됩니다. W&B 에이전트는 컨트롤러와 통신하여 트레이닝에 사용할 다음 파라미터 집합을 결정합니다. 컨트롤러는 또한 조기 종료 알고리즘을 실행하여 어떤 run을 중단할 수 있을지 판단하는 역할도 합니다.

로컬 컨트롤러 기능을 사용하면 사용자가 탐색(search) 및 중단(stop) 알고리즘을 로컬에서 실행할 수 있습니다. 로컬 컨트롤러는 사용자가 코드를 확인하고 조작하여 문제를 디버깅하거나, 클라우드 서비스에 통합할 수 있는 새로운 기능을 개발할 수 있도록 해줍니다.

{{% alert color="secondary" %}}
이 기능은 Sweeps 툴에서 새로운 알고리즘을 더 빠르게 개발하고 디버깅할 수 있도록 지원하기 위한 것입니다. 실제 하이퍼파라미터 최적화 작업에는 권장되지 않습니다.
{{% /alert %}}

시작하기 전에 W&B SDK(`wandb`)를 설치해야 합니다. 다음 코드조각을 커맨드라인에 입력하세요:

```
pip install wandb sweeps 
```

아래 예제들은 이미 설정 파일과 트레이닝 루프가 Python 스크립트나 Jupyter Notebook에 정의되어 있다고 가정합니다. 설정 파일 정의법이 궁금하다면 [스윕 구성 정의하기]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ko" >}})를 참고하세요.

### 커맨드라인에서 로컬 컨트롤러 실행하기

W&B가 클라우드 서비스로 호스팅하는 하이퍼파라미터 컨트롤러를 사용할 때와 비슷한 방식으로 스윕을 초기화하세요. 로컬 컨트롤러를 W&B 스윕 작업에 사용하려면 컨트롤러 플래그(`controller`)를 지정하세요:

```bash
wandb sweep --controller config.yaml
```

또는, 스윕 초기화와 로컬 컨트롤러 사용 지정 단계를 분리할 수도 있습니다.

단계를 분리하려면, 먼저 스윕의 YAML 설정 파일에 아래 키-값을 추가하세요:

```yaml
controller:
  type: local
```

그 다음, 스윕을 초기화합니다:

```bash
wandb sweep config.yaml
```

`wandb sweep` 명령은 스윕 ID를 생성합니다. 스윕을 초기화한 후, [`wandb controller`]({{< relref path="/ref/python/sdk/functions/controller.md" lang="ko" >}})로 컨트롤러를 시작하세요:

```bash
wandb controller {user}/{entity}/{sweep_id}
```

로컬 컨트롤러를 사용하도록 지정했으면, 스윕 실행을 위해 한 개 이상의 Sweep 에이전트를 시작하세요. W&B Sweep을 기존과 같이 시작하면 됩니다. 자세한 내용은 [스윕 에이전트 시작하기]({{< relref path="/guides/models/sweeps/start-sweep-agents.md" lang="ko" >}})를 참고하세요.

```bash
wandb sweep sweep_ID
```

### W&B Python SDK로 로컬 컨트롤러 실행하기

아래 코드조각에서는 W&B Python SDK에서 로컬 컨트롤러를 지정하고 사용하는 방법을 보여줍니다.

Python SDK에서 컨트롤러를 사용하는 가장 간단한 방법은 [`wandb.controller`]({{< relref path="/ref/python/sdk/functions/controller.md" lang="ko" >}}) 메소드에 스윕 ID를 넘기는 것입니다. 그리고 리턴되는 오브젝트의 `run` 메소드를 사용해 스윕 작업을 시작하세요:

```python
sweep = wandb.controller(sweep_id)
sweep.run()
```

컨트롤러 루프를 직접 제어하고 싶을 때는:

```python
import wandb

sweep = wandb.controller(sweep_id)
while not sweep.done():
    sweep.print_status() # 상태 출력
    sweep.step()         # 다음 스텝 진행
    time.sleep(5)        # 5초 대기
```

파라미터 전달까지 더 세밀하게 제어하려면:

```python
import wandb

sweep = wandb.controller(sweep_id)
while not sweep.done():
    params = sweep.search()     # 파라미터 탐색
    sweep.schedule(params)      # 파라미터 지정
    sweep.print_status()        # 상태 출력
```

코드로만 스윕 전체를 정의하고 싶다면 아래와 같이 할 수 있습니다:

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