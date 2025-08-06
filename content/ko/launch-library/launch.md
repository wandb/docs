---
title: 'launch api

  '
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/_launch.py#L249-L331 >}}

W&B Launch 실험을 실행합니다.

```python
launch(
    api: Api,
    job: Optional[str] = None,
    entry_point: Optional[List[str]] = None,
    version: Optional[str] = None,
    name: Optional[str] = None,
    resource: Optional[str] = None,
    resource_args: Optional[Dict[str, Any]] = None,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    docker_image: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    synchronous: Optional[bool] = (True),
    run_id: Optional[str] = None,
    repository: Optional[str] = None
) -> AbstractRun
```

| 인수 |  |
| :--- | :--- |
|  `job` |  wandb.Job 의 문자열 참조 예: wandb/test/my-job:latest |
|  `api` |  wandb.apis.internal 의 wandb Api 인스턴스 |
|  `entry_point` |  프로젝트 내에서 실행할 엔트리 포인트입니다. 기본값은 wandb URI의 경우 원본 run에서 사용한 entry point, git 저장소 URI의 경우 main.py 입니다. |
|  `version` |  Git 기반 프로젝트의 경우 커밋 해시 또는 브랜치 명입니다. |
|  `name` |  실행할 run 의 이름입니다. |
|  `resource` |  run 실행에 사용할 백엔드입니다. |
|  `resource_args` |  remote 백엔드에서 run 을 실행할 때 사용할 resource 관련 인수입니다. 생성된 launch 설정의 `resource_args` 키에 저장됩니다. |
|  `project` |  실행된 run 을 전송할 대상 Project |
|  `entity` |  실행된 run 을 전송할 대상 Entity |
|  `config` |  run 에 대한 설정이 담긴 사전입니다. "resource_args" 키 하에 resource 관련 인수를 포함할 수도 있습니다. |
|  `synchronous` |  run 이 완료될 때까지 대기할지 여부입니다. 기본값은 True 입니다. 참고로, `synchronous`가 False이고 `backend`가 "local-container"인 경우, 이 메소드는 반환되지만 현재 프로세스는 로컬 run 이 완료될 때까지 종료 시 대기합니다. 현재 프로세스가 중단되면, 이 메소드로 비동기 실행된 run 들도 종료됩니다. `synchronous`가 True이고 run 이 실패하면 현재 프로세스도 에러가 발생합니다. |
|  `run_id` |  run 의 ID (궁극적으로 :name: 필드를 대체함) |
|  `repository` |  원격 레지스트리를 위한 저장소 경로의 문자열 이름 |

#### 예시:

```python
from wandb.sdk.launch import launch

job = "wandb/jobs/Hello World:latest"
params = {"epochs": 5}
# W&B 프로젝트를 실행하고, 재현 가능한 도커 환경을
# 로컬 호스트에 생성합니다
api = wandb.apis.internal.Api()
launch(api, job, parameters=params)
```

| 반환값 |  |
| :--- | :--- |
|  실행된 run 에 대한 정보(예: run ID)를 제공하는 `wandb.launch.SubmittedRun` 인스턴스 |

| 예외 |  |
| :--- | :--- |
|  `wandb.exceptions.ExecutionError` 블로킹 모드에서 실행이 실패한 경우 발생합니다. |