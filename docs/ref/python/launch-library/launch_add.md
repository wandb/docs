# launch_add

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/launch/_launch_add.py#L34-L131' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

W&B Launch 실험을 대기열에 추가합니다. 소스 URI, job 또는 docker_image 중 하나로 가능합니다.

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

| 인수 |  |
| :--- | :--- |
|  `uri` | 실행할 실험의 URI입니다. wandb run URI 또는 Git 저장소 URI가 될 수 있습니다. |
|  `job` | wandb.Job에 대한 문자열 참조, 예: wandb/test/my-job:latest |
|  `config` | run을 위한 설정을 포함하는 사전입니다. 또한 "resource_args"라는 키 하에 리소스별 인수를 포함할 수 있습니다. |
|  `template_variables` | 런 큐를 위한 템플릿 변수의 값을 포함하는 사전입니다. `{"VAR_NAME": VAR_VALUE}` 형식으로 예상됩니다. |
|  `project` | 실행된 run을 보낼 대상 프로젝트 |
|  `entity` | 실행된 run을 보낼 대상 엔티티 |
|  `queue` | run을 대기열에 넣을 큐의 이름 |
|  `priority` | job의 우선순위 수준으로, 1이 가장 높은 우선순위입니다. |
|  `resource` | run을 위한 실행 백엔드: W&B는 "local-container" 백엔드를 기본적으로 제공합니다. |
|  `entry_point` | 프로젝트 내에서 실행할 진입점입니다. wandb URIs의 경우 원래 run에 사용된 진입점을 사용하거나, git 저장소 URIs의 경우 main.py를 기본값으로 사용합니다. |
|  `name` | run을 실행할 때 사용할 이름 |
|  `version` | Git 기반 프로젝트의 경우 커밋 해시 또는 브랜치 이름 |
|  `docker_image` | run에 사용할 Docker 이미지의 이름 |
|  `resource_args` | 원격 백엔드로 run을 실행하기 위한 리소스 관련 인수입니다. 설정된 launch config 내 `resource_args`에 저장됩니다. |
|  `run_id` | 실행된 run의 id를 나타내는 선택적 문자열 |
|  `build` | 선택적 플래그로 기본값은 False이며, 빌드하려면 큐를 설정해야 하며, 이미지를 생성하고, job 아티팩트를 생성하고, 그 job 아티팩트에 대한 참조를 큐에 푸시합니다. |
|  `repository` | 이미지가 레지스트리에 푸시될 때 사용되는 원격 저장소의 이름을 제어하는 선택적 문자열 |
|  `project_queue` | 큐를 위한 프로젝트 이름을 제어하는 선택적 문자열입니다. 주로 프로젝트 범위의 큐와의 호환성을 위해 사용됩니다. |

#### 예시:

```python
from wandb.sdk.launch import launch_add

project_uri = "https://github.com/wandb/examples"
params = {"alpha": 0.5, "l1_ratio": 0.01}
# W&B 프로젝트를 실행하고 로컬 호스트에서 재현 가능한 Docker 환경을 만듭니다.
api = wandb.apis.internal.Api()
launch_add(uri=project_uri, parameters=params)
```

| 반환값 |  |
| :--- | :--- |
|  큐에 등록된 run에 대한 정보를 제공하는 `wandb.api.public.QueuedRun` 인스턴스를 반환합니다. 또는 `wait_until_started`나 `wait_until_finished`가 호출되면, 기본 Run 정보에 접근할 수 있습니다. |

| 발생 |  |
| :--- | :--- |
|  실행에 실패한 경우 `wandb.exceptions.LaunchError` 를 발생시킵니다. |