---
title: launch_add
menu:
  reference:
    identifier: ko-ref-python-launch-library-launch_add
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/launch/_launch_add.py#L34-L131 >}}

W&B Launch 실험을 대기열에 추가합니다. source uri, job 또는 docker_image 중 하나를 사용합니다.

```python
launch_add(
    uri: Optional[str] = None,
    job: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    template_variables: Optional[Dict[str, Union[float, int, str]]] = None,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    queue_name: Optional[str] = None,
    resource: Optional[str] = None,
    entry_point: Optional[List[str]] = None,
    name: Optional[str] = None,
    version: Optional[str] = None,
    docker_image: Optional[str] = None,
    project_queue: Optional[str] = None,
    resource_args: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
    build: Optional[bool] = (False),
    repository: Optional[str] = None,
    sweep_id: Optional[str] = None,
    author: Optional[str] = None,
    priority: Optional[int] = None
) -> "public.QueuedRun"
```

| 인Arguments수 |  |
| :--- | :--- |
|  `uri` |  실행할 실험의 URI. wandb run uri 또는 Git 저장소 URI입니다. |
|  `job` |  wandb.Job 에 대한 문자열 참조 (예: wandb/test/my-job:latest) |
|  `config` |  run에 대한 설정을 포함하는 사전입니다. "resource_args" 키 아래에 리소스 특정 인수를 포함할 수도 있습니다. |
|  `template_variables` |  run queue에 대한 템플릿 변수의 값을 포함하는 사전입니다. `{"VAR_NAME": VAR_VALUE}` 형식으로 예상됩니다. |
|  `project` |  실행된 run을 보낼 대상 project |
|  `entity` |  실행된 run을 보낼 대상 entity |
|  `queue` |  run을 대기열에 추가할 queue의 이름 |
|  `priority` |  작업의 우선순위 수준입니다. 1이 가장 높은 우선순위입니다. |
|  `resource` |  run에 대한 실행 백엔드: W&B는 "local-container" 백엔드에 대한 기본 지원을 제공합니다. |
|  `entry_point` |  project 내에서 실행할 entry point. wandb URI의 경우 원래 run에서 사용된 entry point를 사용하거나 git 저장소 URI의 경우 main.py를 기본적으로 사용합니다. |
|  `name` |  run을 시작할 run 이름입니다. |
|  `version` |  Git 기반 프로젝트의 경우 커밋 해시 또는 branch 이름입니다. |
|  `docker_image` |  run에 사용할 docker 이미지의 이름입니다. |
|  `resource_args` |  원격 백엔드에 run을 시작하기 위한 리소스 관련 인수입니다. `resource_args` 아래의 구성된 launch config에 저장됩니다. |
|  `run_id` |  시작된 run의 ID를 나타내는 선택적 문자열입니다. |
|  `build` |  선택적 플래그로 기본값은 false입니다. build인 경우 queue를 설정해야 합니다. 이미지가 생성되고 job Artifacts가 생성되며 해당 job Artifacts에 대한 참조가 queue로 푸시됩니다. |
|  `repository` |  이미지를 레지스트리로 푸시할 때 사용되는 원격 저장소의 이름을 제어하는 선택적 문자열입니다. |
|  `project_queue` |  queue에 대한 project 이름을 제어하는 선택적 문자열입니다. 주로 project 범위 queue와의 이전 버전과의 호환성에 사용됩니다. |

#### 예제:

```python
from wandb.sdk.launch import launch_add

project_uri = "https://github.com/wandb/examples"
params = {"alpha": 0.5, "l1_ratio": 0.01}
# W&B 프로젝트를 실행하고 로컬 호스트에서 재현 가능한 docker 환경을 만듭니다.
api = wandb.apis.internal.Api()
launch_add(uri=project_uri, parameters=params)
```

| 반환 값 |  |
| :--- | :--- |
|  queued run에 대한 정보를 제공하는 `wandb.api.public.QueuedRun` 인스턴스이거나, `wait_until_started` 또는 `wait_until_finished`가 호출된 경우 기본 Run 정보에 대한 엑세스를 제공합니다. |

| 예외 |  |
| :--- | :--- |
|  실패한 경우 `wandb.exceptions.LaunchError` |
