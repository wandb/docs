---
title: 'launch_add

  '
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/_launch_add.py#L34-L131 >}}

W&B Launch experiment 를 대기열에 추가합니다. source uri, job, docker_image 중 하나로 실행할 수 있습니다.

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

| 인수 | 설명 |
| :--- | :--- |
|  `uri` |  실행할 experiment 의 URI 입니다. wandb run uri 또는 Git 저장소 URI 를 사용할 수 있습니다. |
|  `job` |  wandb.Job 을 참조하는 문자열입니다. 예: wandb/test/my-job:latest |
|  `config` |  run 을 위한 설정을 담은 사전입니다. "resource_args"라는 키 아래에 자원 관련 인수가 포함될 수도 있습니다. |
|  `template_variables` |  run 큐를 위한 템플릿 변수의 값을 담고 있는 사전입니다. `{"VAR_NAME": VAR_VALUE}` 형식이 필요합니다. |
|  `project` |  실행된 run 을 보낼 대상 프로젝트입니다. |
|  `entity` |  실행된 run 을 보낼 대상 entity 입니다. |
|  `queue` |  run 을 대기열에 추가할 queue 의 이름입니다. |
|  `priority` |  job 의 우선 순위 레벨입니다. 1이 가장 높은 우선 순위입니다. |
|  `resource` |  run 을 실행할 백엔드입니다. W&B 는 "local-container" 백엔드를 기본 지원합니다. |
|  `entry_point` |  프로젝트 내에서 실행할 entry point 입니다. wandb URI 는 원본 run 의 entry point 를, git 저장소 URI 는 기본적으로 main.py 를 사용합니다. |
|  `name` |  실행할 run 의 이름입니다. |
|  `version` |  Git 기반 프로젝트용으로, 커밋 해시 또는 브랜치 이름을 지정할 수 있습니다. |
|  `docker_image` |  run 에 사용할 docker 이미지 이름입니다. |
|  `resource_args` |  원격 백엔드에 run 을 실행할 때 필요한 자원 관련 인수입니다. 이 값은 생성된 launch config 의 `resource_args`에 저장됩니다. |
|  `run_id` |  실행된 run 의 id 를 나타내는 선택적 문자열입니다. |
|  `build` |  선택적 플래그이며 기본값은 False 입니다. build 가 true 이면 queue 설정이 필요하며, 이미지를 생성할 때 job artifact 를 만들고, 이 job artifact 의 참조를 queue 에 추가합니다. |
|  `repository` |  레지스트리에 이미지를 푸시할 때 사용할 원격 저장소의 이름을 제어하는 선택적 문자열입니다. |
|  `project_queue` |  큐에 사용할 프로젝트의 이름을 제어하는 선택적 문자열입니다. 주로 프로젝트 범위 큐의 하위 호환성을 위해 사용됩니다. |

#### 예시:

```python
from wandb.sdk.launch import launch_add

project_uri = "https://github.com/wandb/examples"
params = {"alpha": 0.5, "l1_ratio": 0.01}
# W&B 프로젝트를 실행하고 재현 가능한 docker 환경을
# 로컬 호스트에서 생성합니다.
api = wandb.apis.internal.Api()
launch_add(uri=project_uri, parameters=params)
```

| 반환 값 | 설명 |
| :--- | :--- |
|  `wandb.api.public.QueuedRun` 인스턴스를 반환하며, 큐에 추가된 run 에 대한 정보를 제공합니다. 또는 `wait_until_started`나 `wait_until_finished`를 호출하면, 기본 Run 정보를 엑세스할 수 있습니다. |

| 예외 | 설명 |
| :--- | :--- |
|  실패시 `wandb.exceptions.LaunchError` 발생 |