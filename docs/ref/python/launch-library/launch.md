# launch

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/launch/_launch.py#L248-L330' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

W&B launch 실험을 시작합니다.

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
|  `job` |  wandb.Job에 대한 문자열 참조 예: wandb/test/my-job:latest |
|  `api` |  wandb.apis.internal에서 가져온 wandb Api 인스턴스. |
|  `entry_point` |  프로젝트 내에서 실행할 시작점입니다. wandb URI에서는 원래 실행에 사용된 entry_point를 기본값으로 사용하며, git repository URI에서는 main.py가 기본값입니다. |
|  `version` |  Git 기반 프로젝트의 경우, 커밋 해시 또는 브랜치 이름입니다. |
|  `name` |  실행할 run의 이름입니다. |
|  `resource` |  run을 실행할 백엔드입니다. |
|  `resource_args` |  원격 백엔드에서 run을 시작하기 위한 리소스 관련 인수입니다. 이는 구성된 launch 설정의 `resource_args` 아래에 저장될 것입니다. |
|  `project` |  실행된 run을 보낼 대상 프로젝트입니다. |
|  `entity` |  실행된 run을 보낼 대상 entity입니다. |
|  `config` |  run 설정을 포함하는 사전입니다. 또한 "resource_args"라는 키 아래에 리소스 관련 인수를 포함할 수 있습니다. |
|  `synchronous` |  run 완료 대기 중에 블록할지 여부입니다. 기본값은 True입니다. 만약 `synchronous`가 False이고 `backend`가 "local-container"인 경우, 이 메소드는 반환되지만 현재 프로세스는 로컬 run이 완료될 때까지 종료 시 블록합니다. 현재 프로세스가 중단되면 이 메소드를 통해 시작된 비동기 run은 종료됩니다. `synchronous`가 True이고 run이 실패하면, 현재 프로세스도 오류가 발생합니다. |
|  `run_id` | run의 ID입니다 (궁극적으로 :name: 필드를 대체하기 위함) |
|  `repository` |  원격 레지스트리의 repository 경로에 대한 문자열 이름 |

#### 예시:

```python
from wandb.sdk.launch import launch

job = "wandb/jobs/Hello World:latest"
params = {"epochs": 5}
# W&B 프로젝트를 실행하고 로컬 호스트에서
# 재현 가능한 도커 환경을 생성합니다.
api = wandb.apis.internal.Api()
launch(api, job, parameters=params)
```

| 반환값 |  |
| :--- | :--- |
|  실행된 run에 대한 정보(예: run ID)를 노출하는 `wandb.launch.SubmittedRun` 인스턴스입니다. |

| 예외 |  |
| :--- | :--- |
|  `wandb.exceptions.ExecutionError` 블록 모드에서 실행된 run이 성공적이지 않은 경우. |