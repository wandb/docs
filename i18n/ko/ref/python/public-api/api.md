
# Api

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/public/api.py#L98-L1044' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>


wandb 서버를 쿼리하는 데 사용됩니다.

```python
Api(
    overrides=None,
    timeout: Optional[int] = None,
    api_key: Optional[str] = None
) -> None
```

#### 예제:

가장 일반적인 초기화 방법

```
>>> wandb.Api()
```

| 인수 |  |
| :--- | :--- |
|  `overrides` |  (dict) https://api.wandb.ai가 아닌 다른 wandb 서버를 사용하는 경우 `base_url`을 설정할 수 있습니다. 또한 `entity`, `project`, 그리고 `run`의 기본값을 설정할 수 있습니다. |

| 속성 |  |
| :--- | :--- |

## 메소드

### `artifact`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/public/api.py#L941-L965)

```python
artifact(
    name, type=None
)
```

`entity/project/name` 형식의 경로를 구문 분석하여 단일 아티팩트를 반환합니다.

| 인수 |  |
| :--- | :--- |
|  `name` |  (str) 아티팩트 이름. entity/project로 접두사가 붙을 수 있습니다. 유효한 이름은 다음 형식일 수 있습니다: name:version name:alias |
|  `type` |  (str, optional) 가져올 아티팩트의 타입입니다. |

| 반환값 |  |
| :--- | :--- |
|  `Artifact` 오브젝트. |

### `artifact_collection`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/public/api.py#L910-L924)

```python
artifact_collection(
    type_name: str,
    name: str
)
```

타입과 `entity/project/name` 형식의 경로를 구문 분석하여 단일 아티팩트 컬렉션을 반환합니다.

| 인수 |  |
| :--- | :--- |
|  `name` |  (str) 아티팩트 컬렉션 이름. entity/project로 접두사가 붙을 수 있습니다. |
|  `type` |  (str) 가져올 아티팩트 컬렉션의 타입입니다. |

| 반환값 |  |
| :--- | :--- |
|  `ArtifactCollection` 오브젝트. |

### `artifact_collections`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/public/api.py#L903-L908)

```python
artifact_collections(
    project_name: str,
    type_name: str,
    per_page=50
)
```

### `artifact_type`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/public/api.py#L898-L901)

```python
artifact_type(
    type_name, project=None
)
```

### `artifact_types`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/public/api.py#L893-L896)

```python
artifact_types(
    project=None
)
```

### `artifact_versions`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/public/api.py#L926-L932)

```python
artifact_versions(
    type_name, name, per_page=50
)
```

사용 중단됨, 대신 artifacts(type_name, name)를 사용하세요.

### `artifacts`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/public/api.py#L934-L939)

```python
artifacts(
    type_name, name, per_page=50
)
```

### `create_project`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/public/api.py#L273-L274)

```python
create_project(
    name: str,
    entity: str
)
```

### `create_report`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/public/api.py#L282-L297)

```python
create_report(
    project: str,
    entity: str = "",
    title: Optional[str] = "무제 리포트",
    description: Optional[str] = "",
    width: Optional[str] = "readable",
    blocks: Optional['wandb.apis.reports.util.Block'] = None
) -> "wandb.apis.reports.Report"
```

### `create_run`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/public/api.py#L276-L280)

```python
create_run(
    **kwargs
)
```

새 run을 생성합니다.

### `create_run_queue`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/public/api.py#L299-L409)

```python
create_run_queue(
    name: str,
    type: "public.RunQueueResourceType",
    entity: Optional[str] = None,
    prioritization_mode: Optional['public.RunQueuePrioritizationMode'] = None,
    config: Optional[dict] = None,
    template_variables: Optional[dict] = None
) -> "public.RunQueue"
```

새 run 큐(런치)를 생성합니다.

| 인수 |  |
| :--- | :--- |
|  `name` |  (str) 생성할 큐의 이름 |
|  `type` |  (str) 큐에 사용될 자원의 타입. "local-container", "local-process", "kubernetes", "sagemaker", 또는 "gcp-vertex" 중 하나입니다. |
|  `entity` |  (str) 큐를 생성할 엔티티의 선택적 이름. None인 경우, 설정된 또는 기본 엔티티를 사용합니다. |
|  `prioritization_mode` |  (str) 사용할 우선순위 모드의 선택적 버전. "V0" 또는 None입니다. |
|  `config` |  (dict) 큐에 사용될 선택적 기본 자원 설정. 구성 변수를 지정하기 위해 핸들바(예: "{{var}}")를 사용합니다. template_variables (dict): config와 함께 사용할 템플릿 변수 스키마의 딕셔너리. 예상 형식: { "var-name": { "schema": { "type": "<string | number | integer>", "default": <optional value>, "minimum": <optional minimum>, "maximum": <optional maximum>, "enum": [..."<options>"] } } } |

| 반환값 |  |
| :--- | :--- |
|  새로 생성된 `RunQueue` |

| 예외 |  |
| :--- | :--- |
|  파라미터 중 하나라도 유효하지 않은 경우 ValueError wandb API 오류가 발생하면 wandb.Error |

### `create_team`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/public/api.py#L690-L700)

```python
create_team(
    team, admin_username=None
)
```

새 팀을 생성합니다.

| 인수 |  |
| :--- | :--- |
|  `team` |  (str) 팀의 이름 |
|  `admin_username` |  (str) 팀의 관리자 사용자 이름, 기본값은 현재 사용자입니다. |

| 반환값 |  |
| :--- | :--- |
|  `Team` 오브젝트 |

### `create_user`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/public/api.py#L427-L437)

```python
create_user(
    email, admin=(False)
)
```

새 사용자를 생성합니다.

| 인수 |  |
| :--- | :--- |
|  `email` |  (str) 사용자의 이메일 주소 |
|  `admin` |  (bool) 이 사용자가 전역 인스턴스 관리자가 되어야 하는지 여부 |

| 반환값 |  |
| :--- | :--- |
|  `User` 오브젝트 |

### `flush`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/public/api.py#L504-L511)

```python
flush()
```

로컬 캐시를 비웁니다.

api 오브젝트는 run의 로컬 캐시를 유지하므로 스크립트 실행 중 run의 상태가 변경될 수 있다면 `api.flush()`로 로컬 캐시를 지워 run과 관련된 최신 값들을 얻어야 합니다.

### `from_path`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/public/api.py#L513-L567)

```python
from_path(
    path
)
```

경로에서 run, 스윕, 프로젝트 또는 리포트를 반환합니다.

#### 예제:

```
project = api.from_path("my_project")
team_project = api.from_path("my_team/my_project")
run = api.from_path("my_team/my_project/runs/id")
sweep = api.from_path("my_team/my_project/sweeps/id")
report = api.from_path("my_team/my_project/reports/My-Report-Vm11dsdf")
```

| 인수 |  |
| :--- | :--- |
|  `path` |  (str) 프로젝트, run, 스윕 또는 리포트로의 경로 |

| 반환값 |  |
| :--- | :--- |
|  `Project`, `Run`, `Sweep`, 또는 `BetaReport` 인스턴스. |

| 예외 |  |
| :--- | :--- |
|  경로가 유효하지 않거나 오브젝트가 존재하지 않으면 wandb.Error |

### `job`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/public/api.py#L967-L975)

```python
job(
    name, path=None
)
```

### `list_jobs`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/public/api.py#L977-L1044)

```python
list_jobs(
    entity, project
)
```

### `load_report`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/public/api.py#L411-L425)

```python
load_report(
    path: str
) -> "wandb.apis.reports.Report"
```

주어진 경로의 리포트를 가져옵니다.

| 인수 |  |
| :--- | :--- |
|  `path` |  (str) `entity/project/reports/reportId` 형식의 대상 리포트로의 경로. 이것은 wandb URL 다음의 URL을 복사하여 붙여넣음으로써 얻을 수 있습니다. 예: `megatruong/report-editing/reports/My-fabulous-report-title--VmlldzoxOTc1Njk0` |

| 반환값 |  |
| :--- | :--- |
|  `path`에 있는 리포트를 나타내는 `BetaReport` 오브젝트 |

| 예외 |  |
| :--- | :--- |
|  경로가 유효하지 않으면 wandb.Error |

### `project`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/public/api.py#L654-L657)

```python
project(
    name, entity=None
)
```

### `projects`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/public/api.py#L629-L652)

```python
projects(
    entity=None, per_page=200
)
```

주어진 엔티티의 프로젝트를 가져옵니다.

| 인수 |  |
| :--- | :--- |
|  `entity` |  (str) 요청된 엔티티의 이름. None인 경우, `Api`에 전달된 기본 엔티티로 대체됩니다. 기본 엔티티가 없는 경우 `ValueError`를 발생시킵니다. |
|  `per_page` |  (int) 쿼리 페이지네이션의 페이지 크기를 설정합니다. None은 기본 크기를 사용합니다. 일반적으로 이것을 변경할 필요는 없습니다. |

| 반환값 |  |
| :--- | :--- |
|  `Project` 오브젝트의 반복 가능한 컬렉션인 `Projects` 오브젝트. |

### `queued_run`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/public/api.py#L838-L859)

```python
queued_run(
    entity, project, queue_name, run_queue_item_id, project_queue=None,
    priority=None
)
```

경로를 기반으로 단일 대기열 run을 반환합니다.

`entity/project/queue_id/run_queue_item_id` 형식의 경로를 구문 분석합니다.

### `reports`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/public/api.py#L659-L688)

```python
reports(
    path="", name=None, per_page=50
)
```

주어진 프로젝트 경로에 대한 리포트를 가져옵니다.

경고: 이 API는 베타 버전이며 향후 릴리스에서 변경될 가능성이 있습니다.

| 인수 |  |
| :--- | :--- |
|  `path` |  (str) 리포트가 속한 프로젝트로의 경로, 형식은 "entity/project"여야 합니다. |
|  `name` |  (str) 요청된 리포트의 선택적 이름. |
|  `per_page` |  (int) 쿼리 페이지네이션의 페이지 크기를 설정합니다. None은 기본 크기를 사용합니다. 일반적으로 이것을 변경할 필요는 없습니다. |

| 반환값 |  |
| :--- | :--- |
|  `BetaReport` 오브젝트의 반복 가능한 컬렉션인 `Reports` 오브젝트. |

### `run`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/public/api.py#L821-L836)

```python
run(
    path=""
)
```

`entity/project/run_id` 형식의 경로를 구문 분석하여 단일 run을 반환합니다.

| 인수 |  |
| :--- | :--- |
|  `path` |  (str) `entity/project/run_id` 형식으로의 run 경로. `api.entity`가 설정되어 있다면, 이것은 `project/run_id` 형식일 수 있고, `api.project`가 설정되어 있다면 이것은 단순히 run_id일 수 있습니다. |

| 반환값 |  |
| :--- | :--- |
|  `Run` 오브젝트. |

### `run_queue`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/public/api.py#L861-L874)

```python
run_queue(
    entity, name
)
```

엔티티에 대한 명명된 `RunQueue`를 반환합니다.

새 `RunQueue`를 생성하려면