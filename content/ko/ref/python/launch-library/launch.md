---
title: launch
menu:
  reference:
    identifier: ko-ref-python-launch-library-launch
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/launch/_launch.py#L249-L331 >}}

W&B Launch 실험을 시작합니다.

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

| 인자 |  |
| :--- | :--- |
|  `job` |  wandb.Job 에 대한 문자열 참조 (예: wandb/test/my-job:latest) |
|  `api` |  wandb.apis.internal 의 wandb Api 인스턴스입니다. |
|  `entry_point` |  프로젝트 내에서 실행할 엔트리 포인트입니다. wandb URI의 경우 원래 run에 사용된 엔트리 포인트를 사용하고, Git repository URI의 경우 main.py를 사용하는 것이 기본값입니다. |
|  `version` |  Git 기반 프로젝트의 경우 커밋 해시 또는 브랜치 이름입니다. |
|  `name` |  run을 시작할 run 이름입니다. |
|  `resource` |  run을 위한 실행 백엔드입니다. |
|  `resource_args` |  원격 백엔드에서 run을 시작하기 위한 리소스 관련 인자입니다. `resource_args` 아래 생성된 launch 설정에 저장됩니다. |
|  `project` |  시작된 run을 보낼 대상 프로젝트 |
|  `entity` |  시작된 run을 보낼 대상 엔티티 |
|  `config` |  run에 대한 설정을 포함하는 사전입니다. "resource_args" 키 아래에 리소스 특정 인자를 포함할 수도 있습니다. |
|  `synchronous` |  run이 완료될 때까지 기다리는 동안 차단할지 여부입니다. 기본값은 True입니다. `synchronous`가 False이고 `backend`가 "local-container"인 경우 이 메소드는 반환되지만 로컬 run이 완료될 때까지 현재 프로세스는 종료 시 차단됩니다. 현재 프로세스가 중단되면 이 메소드를 통해 시작된 모든 비동기 run이 종료됩니다. `synchronous`가 True이고 run이 실패하면 현재 프로세스도 오류가 발생합니다. |
|  `run_id` |  run의 ID (:name: 필드를 궁극적으로 대체하기 위해) |
|  `repository` |  원격 레지스트리의 repository 경로의 문자열 이름 |

#### 예시:

```python
from wandb.sdk.launch import launch

job = "wandb/jobs/Hello World:latest"
params = {"epochs": 5}
# W&B 프로젝트를 실행하고 로컬 호스트에서 재현 가능한 도커 환경을 만듭니다.
api = wandb.apis.internal.Api()
launch(api, job, parameters=params)
```

| 반환 값 |  |
| :--- | :--- |
|  시작된 run에 대한 정보 (예: run ID)를 노출하는 `wandb.launch.SubmittedRun`의 인스턴스입니다. |

| 예외 발생 |  |
| :--- | :--- |
|  `wandb.exceptions.ExecutionError` 차단 모드에서 시작된 run이 실패한 경우. |
