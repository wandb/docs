# Api

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/api.py#L95-L1179' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

wandb 서버에 쿼리할 때 사용됩니다.

```python
Api(
    overrides: Optional[Dict[str, Any]] = None,
    timeout: Optional[int] = None,
    api_key: Optional[str] = None
) -> None
```

#### 예시:

가장 일반적인 초기화 방법

```
>>> wandb.Api()
```

| 인수 |  |
| :--- | :--- |
| `overrides` | (dict) https://api.wandb.ai가 아닌 다른 wandb 서버를 사용하는 경우 `base_url`을 설정할 수 있습니다. 또한, `entity`, `project`, 그리고 `run`의 기본값을 설정할 수도 있습니다. |

| 속성 |  |
| :--- | :--- |

## 메소드

### `artifact`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/api.py#L1017-L1041)

```python
artifact(
    name, type=None
)
```

`entity/project/name` 형식의 경로를 분석하여 단일 아티팩트를 반환합니다.

| 인수 |  |
| :--- | :--- |
| `name` | (str) 아티팩트 이름입니다. entity/project로 접두어가 붙을 수 있습니다. 유효한 이름 형식: name:version name:alias |
| `type` | (str, optional) 가져올 아티팩트의 유형입니다. |

| 반환값 |  |
| :--- | :--- |
| `Artifact` 오브젝트. |

### `artifact_collection`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/api.py#L967-L983)

```python
artifact_collection(
    type_name: str,
    name: str
) -> "public.ArtifactCollection"
```

유형 및 경로 분석을 통해 단일 아티팩트 컬렉션을 반환합니다. 형식은 `entity/project/name`입니다.

| 인수 |  |
| :--- | :--- |
| `type_name` | (str) 가져올 아티팩트 컬렉션의 유형입니다. |
| `name` | (str) 아티팩트 컬렉션 이름입니다. entity/project로 접두어가 붙을 수 있습니다. |

| 반환값 |  |
| :--- | :--- |
| `ArtifactCollection` 오브젝트. |

### `artifact_collection_exists`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/api.py#L1162-L1179)

```python
artifact_collection_exists(
    name: str,
    type: str
) -> bool
```

지정된 프로젝트와 entity 내에서 아티팩트 컬렉션이 존재하는지 여부를 반환합니다.

| 인수 |  |
| :--- | :--- |
| `name` | (str) 아티팩트 컬렉션 이름입니다. entity/project로 접두어가 붙을 수 있습니다. entity나 project가 지정되지 않으면, 삽입된 매개변수에서 추론됩니다. 그렇지 않으면 entity는 사용자 설정에서 가져오고 project는 "uncategorized"로 기본 설정됩니다. |
| `type` | (str) 아티팩트 컬렉션 유형 |

| 반환값 |  |
| :--- | :--- |
| 아티팩트 컬렉션이 존재하면 True, 그렇지 않으면 False를 반환합니다. |

### `artifact_collections`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/api.py#L947-L965)

```python
artifact_collections(
    project_name: str,
    type_name: str,
    per_page: Optional[int] = 50
) -> "public.ArtifactCollections"
```

일치하는 아티팩트 컬렉션의 모음을 반환합니다.

| 인수 |  |
| :--- | :--- |
| `project_name` | (str) 필터링할 프로젝트 이름입니다. |
| `type_name` | (str) 필터링할 아티팩트 유형의 이름입니다. |
| `per_page` | (int, optional) 쿼리 페이징의 페이지 크기를 설정합니다. None이면 기본 크기를 사용합니다. 보통 이를 변경할 이유는 없습니다. |

| 반환값 |  |
| :--- | :--- |
| 반복 가능한 `ArtifactCollections` 오브젝트. |

### `artifact_exists`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/api.py#L1140-L1160)

```python
artifact_exists(
    name: str,
    type: Optional[str] = None
) -> bool
```

지정된 프로젝트와 entity 내에서 아티팩트 버전이 존재하는지 여부를 반환합니다.

| 인수 |  |
| :--- | :--- |
| `name` | (str) 아티팩트 이름입니다. entity/project로 접두어가 붙을 수 있습니다. entity나 project가 지정되지 않으면, 삽입된 매개변수에서 추론됩니다. 그렇지 않으면 entity는 사용자 설정에서 가져오고 project는 "uncategorized"로 기본 설정됩니다. 유효한 이름 형식: name:version name:alias |
| `type` | (str, optional) 아티팩트 유형 |

| 반환값 |  |
| :--- | :--- |
| 아티팩트 버전이 존재하면 True, 그렇지 않으면 False를 반환합니다. |

### `artifact_type`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/api.py#L931-L945)

```python
artifact_type(
    type_name: str,
    project: Optional[str] = None
) -> "public.ArtifactType"
```

일치하는 `ArtifactType`을 반환합니다.

| 인수 |  |
| :--- | :--- |
| `type_name` | (str) 검색할 아티팩트 유형의 이름입니다. |
| `project` | (str, optional) 주어졌다면, 필터링할 프로젝트 이름이나 경로입니다. |

| 반환값 |  |
| :--- | :--- |
| `ArtifactType` 오브젝트. |

### `artifact_types`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/api.py#L918-L929)

```python
artifact_types(
    project: Optional[str] = None
) -> "public.ArtifactTypes"
```

일치하는 아티팩트 유형의 모음을 반환합니다.

| 인수 |  |
| :--- | :--- |
| `project` | (str, optional) 주어졌다면, 필터링할 프로젝트 이름이나 경로입니다. |

| 반환값 |  |
| :--- | :--- |
| 반복 가능한 `ArtifactTypes` 오브젝트. |

### `artifact_versions`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/api.py#L985-L995)

```python
artifact_versions(
    type_name, name, per_page=50
)
```

폐기 예정, 대신 `artifacts(type_name, name)`을 사용하세요.

### `artifacts`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/api.py#L997-L1015)

```python
artifacts(
    type_name: str,
    name: str,
    per_page: Optional[int] = 50
) -> "public.Artifacts"
```

주어진 인수로부터 `Artifacts` 컬렉션을 반환합니다.

| 인수 |  |
| :--- | :--- |
| `type_name` | (str) 가져올 아티팩트 유형. |
| `name` | (str) 아티팩트 컬렉션 이름입니다. entity/project로 접두어가 붙을 수 있습니다. |
| `per_page` | (int, optional) 쿼리 페이징의 페이지 크기를 설정합니다. None이면 기본 크기를 사용합니다. 보통 이를 변경할 이유는 없습니다. |

| 반환값 |  |
| :--- | :--- |
| 반복 가능한 `Artifacts` 오브젝트. |

### `create_project`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/api.py#L281-L288)

```python
create_project(
    name: str,
    entity: str
) -> None
```

새 프로젝트를 만듭니다.

| 인수 |  |
| :--- | :--- |
| `name` | (str) 새 프로젝트의 이름. |
| `entity` | (str) 새 프로젝트의 entity. |

### `create_run`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/api.py#L290-L310)

```python
create_run(
    *,
    run_id: Optional[str] = None,
    project: Optional[str] = None,
    entity: Optional[str] = None
) -> "public.Run"
```

새로운 run을 만듭니다.

| 인수 |  |
| :--- | :--- |
| `run_id` | (str, optional) run에 할당될 ID입니다. run ID는 기본적으로 자동 생성되므로, 일반적으로 지정할 필요가 없으며, 지정 시에는 신중해야 합니다. |
| `project` | (str, optional) 주어졌다면, 새로운 run의 프로젝트입니다. |
| `entity` | (str, optional) 주어졌다면, 새로운 run의 entity입니다. |

| 반환값 |  |
| :--- | :--- |
| 새로 생성된 `Run`. |

### `create_run_queue`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/api.py#L312-L422)

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

새로운 run 큐를 생성합니다 (Launch).

| 인수 |  |
| :--- | :--- |
| `name` | (str) 생성할 큐 이름 |
| `type` | (str) 큐에 사용할 자원의 유형. "local-container", "local-process", "kubernetes", "sagemaker", "gcp-vertex" 중 하나. |
| `entity` | (str) 큐를 생성할 entity의 선택적 이름. None이면 기본으로 구성된 entity를 사용합니다. |
| `prioritization_mode` | (str) 선택적 우선순위 버전 사용. "V0" 또는 None. |
| `config` | (dict) 큐에 사용할 기본 자원 설정. 핸들바(예: `{{var}}`)로 템플릿 변수를 지정하세요. |
| `template_variables` | (dict) config와 함께 사용할 템플릿 변수 스키마의 사전. 예상 형식: `{ "var-name": { "schema": { "type": ("string", "number", or "integer"), "default": (optional value), "minimum": (optional minimum), "maximum": (optional maximum), "enum": [..."(options)"] }}}` |

| 반환값 |  |
| :--- | :--- |
| 새로 생성된 `RunQueue` |

| 오류 |  |
| :--- | :--- |
| 매개변수가 유효하지 않으면 ValueError를 발생시키고 wandb API 오류가 발생하면 wandb.Error를 발생시킵니다. |

### `create_team`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/api.py#L705-L715)

```python
create_team(
    team, admin_username=None
)
```

새로운 팀을 생성합니다.

| 인수 |  |
| :--- | :--- |
| `team` | (str) 팀의 이름 |
| `admin_username` | (str) 팀의 관리자 사용자 이름, 기본값은 현재 사용자입니다. |

| 반환값 |  |
| :--- | :--- |
| `Team` 오브젝트 |

### `create_user`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/api.py#L424-L434)

```python
create_user(
    email, admin=(False)
)
```

새로운 사용자를 생성합니다.

| 인수 |  |
| :--- | :--- |
| `email` | (str) 사용자의 이메일 주소 |
| `admin` | (bool) 이 사용자가 글로벌 인스턴스 관리자여야 하는지 여부 |

| 반환값 |  |
| :--- | :--- |
| `User` 오브젝트 |

### `flush`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/api.py#L501-L508)

```python
flush()
```

로컬 캐시를 플러시합니다.

API 오브젝트는 run의 로컬 캐시를 유지하므로, 스크립트를 실행하는 동안 run의 상태가 변경될 수 있는 경우에는 `api.flush()`를 사용하여 run과 관련된 최신 값을 가져와야 합니다.

### `from_path`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/api.py#L510-L564)

```python
from_path(
    path
)
```

경로에서 run, sweep, project 또는 report를 반환합니다.

#### 예시:

```
project = api.from_path("my_project")
team_project = api.from_path("my_team/my_project")
run = api.from_path("my_team/my_project/runs/id")
sweep = api.from_path("my_team/my_project/sweeps/id")
report = api.from_path("my_team/my_project/reports/My-Report-Vm11dsdf")
```

| 인수 |  |
| :--- | :--- |
| `path` | (str) 프로젝트, run, sweep 또는 report의 경로 |

| 반환값 |  |
| :--- | :--- |
| `Project`, `Run`, `Sweep`, 또는 `BetaReport` 인스턴스. |

| 오류 |  |
| :--- | :--- |
| 경로가 잘못되었거나 오브젝트가 존재하지 않으면 wandb.Error를 발생시킵니다. |

### `job`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/api.py#L1043-L1060)

```python
job(
    name: Optional[str],
    path: Optional[str] = None
) -> "public.Job"
```

주어진 인수로부터 `Job`을 반환합니다.

| 인수 |  |
| :--- | :--- |
| `name` | (str) 작업의 이름. |
| `path` | (str, optional) 주어졌다면, 작업 아티팩트를 다운로드할 루트 경로입니다. |

| 반환값 |  |
| :--- | :--- |
| `Job` 오브젝트. |

### `list_jobs`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/api.py#L1062-L1138)

```python
list_jobs(
    entity: str,
    project: str
) -> List[Dict[str, Any]]
```

주어진 entity와 프로젝트에 대한 작업 목록을 반환합니다.

| 인수 |  |
| :--- | :--- |
| `entity` | (str) 나열된 작업의 entity. |
| `project` | (str) 나열된 작업의 프로젝트. |

| 반환값 |  |
| :--- | :--- |
| 일치하는 작업 목록. |

### `project`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/api.py#L657-L670)

```python
project(
    name: str,
    entity: Optional[str] = None
) -> "public.Project"
```

주어진 이름 (및 주어졌다면 entity)과 일치하는 `Project`를 반환합니다.

| 인수 |  |
| :--- | :--- |
| `name` | (str) 프로젝트 이름. |
| `entity` | (str) 요청된 entity의 이름입니다. None이면, `Api`에 전달된 기본 entity를 사용합니다. 기본 entity가 없으면 `ValueError`를 발생시킵니다. |

| 반환값 |  |
| :--- | :--- |
| `Project` 오브젝트. |

### `projects`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/api.py#L631-L655)

```python
projects(
    entity: Optional[str] = None,
    per_page: Optional[int] = 200
) -> "public.Projects"
```

주어진 entity에 대한 프로젝트를 가져옵니다.

| 인수 |  |
| :--- | :--- |
| `entity` | (str) 요청된 entity의 이름입니다. None이면, `Api`에 전달된 기본 entity를 사용합니다. 기본 entity가 없으면 `ValueError`를 발생시킵니다. |
| `per_page` | (int) 쿼리 페이징의 페이지 크기를 설정합니다. None이면 기본 크기를 사용합니다. 보통 이를 변경할 이유는 없습니다. |

| 반환값 |  |
| :--- | :--- |
| `Projects` 오브젝트로, `Project` 오브젝트의 반복 가능한 컬렉션입니다. |

### `queued_run`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/api.py#L863-L884)

```python
queued_run(
    entity, project, queue_name, run_queue_item_id, project_queue=None,
    priority=None
)
```

경로를 기반으로 하나의 대기 중인 run을 반환합니다.

경로 형식: entity/project/queue_id/run_queue_item_id을 해석합니다.

### `reports`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/api.py#L672-L703)

```python
reports(
    path: str = "",
    name: Optional[str] = None,
    per_page: Optional[int] = 50
) -> "public.Reports"
```

주어진 프로젝트 경로에 대한 Reports를 가져옵니다.

경고: 이 API는 베타 상태이며, 향후 릴리스에서 변경될 가능성이 있습니다.

| 인수 |  |
| :--- | :--- |
| `path` | (str) 리포트가 위치한 프로젝트의 경로입니다. 형식은 "entity/project"이어야 합니다. |
| `name` | (str, optional) 요청된 리포트의 선택적 이름입니다. |
| `per_page` | (int) 쿼리 페이징의 페이지 크기를 설정합니다. None이면 기본 크기를 사용합니다. 보통 이를 변경할 이유는 없습니다. |

| 반환값 |  |
| :--- | :--- |
| `Reports` 오브젝트로, `BetaReport` 오브젝트의 반복 가능한 컬렉션입니다. |

### `run`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/api.py#L846-L861)

```python
run(
    path=""
)
```

경로를 분석하여 단일 run을 반환합니다. 형식은 entity/project/run_id입니다.

| 인수 |  |
| :--- | :--- |
| `path` | (str) run의 경로로, `entity/project/run_id` 형식을 가져야 합니다. 만약 `api.entity`가 설정되어 있다면, `project/run_id` 형식을 사용할 수 있으며, `api.project`가 설정되어 있다면 run_id만 사용 가능합니다. |

| 반환값 |  |
| :--- | :--- |
| `Run` 오브젝트. |

### `run_queue`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/api.py#L886-L899)

```python
run_queue(
    entity, name
)
```

entity에 대한 지정된 이름의 `RunQueue`를 반환합니다.

새로운 `RunQueue`를 생성하려면 `wandb.Api().create_run_queue(...)`를 사용하세요.

### `runs`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/api.py#L766-L844)

```python
runs(
    path: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    order: str = "+created_at",
    per_page: int = 50,
    include_sweeps: bool = (True)
)
```

제공된 필터에 맞는 프로젝트의 여러 run을 반환합니다.

`config.*`, `summary_metrics.*`, `tags`, `state`, `entity`, `createdAt` 등을 필터링할 수 있습니다.

#### 예시:

config.experiment_name이 "foo"로 설정된 my_project의 run을 찾습니다.

```
api.runs(path="my_entity/my_project", filters={"config.experiment_name": "foo"})
```

config.experiment_name이 "foo" 또는 "bar"로 설정된 my_project의 run을 찾습니다.

```
api.runs(
    path="my_entity/my_project",
    filters={"$or": [{"config.experiment_name": "foo"}, {"config.experiment_name": "bar"}]}
)
```

config.experiment_name이 정규 표현식을 만족하는 my_project의 run을 찾습니다. (앵커는 지원되지 않음)

```
api.runs(
    path="my_entity/my_project",
    filters={"config.experiment_name": {"$regex": "b.*"}}
)
```

run 이름이 정규 표현식을 만족하는 my_project의 run을 찾습니다. (앵커는 지원되지 않음)

```
api.runs(
    path="my_entity/my_project",
    filters={"display_name": {"$regex": "^foo.*"}}
)
```

loss를 기준으로 오름차순으로 정렬된 my_project의 run을 찾습니다.

```
api.runs(path="my_entity/my_project", order="+summary_metrics.loss")
```

| 인수 |  |
| :--- | :--- |
| `path` | (str) 프로젝트 경로, 형식은 "entity/project"이어야 합니다. |
| `filters` | (dict) MongoDB 쿼리 언어를 사용하여 특정 run을 쿼리합니다. `config.key`, `summary_metrics.key`, `state`, `entity`, `createdAt` 등과 같은 run 속성으로 필터링할 수 있습니다. 예: \{"config.experiment_name": "foo"\}는 실험 이름이 "foo"로 설정된 run을 찾습니다. 더 복잡한 쿼리를 작성하려면 작업을 구성할 수 있습니다. 언어에 대한 참조는 https://docs.mongodb.com/manual/reference/operator/query에 있습니다. |
| `order` | (str) 정렬은 `created_at`, `heartbeat_at`, `config.*.value`, 또는 `summary_metrics.*`. 정렬의 접두사로 +를 붙이면 오름차순입니다. 접두사로 -를 붙이면 내림차순(기본값)입니다. 기본 정렬은 run.created_at로 가장 오래된 것부터 가장 최신 것까지입니다. |
| `per_page` | (int) 쿼리 페이징의 페이지 크기를 설정합니다. |
| `include_sweeps` | (bool) 결과에 스윕 run을 포함할지 여부입니다. |

| 반환값 |  |
| :--- | :--- |
| `Runs` 오브젝트로, `Run` 오브젝트의 반복 가능한 컬렉션입니다. |

### `sweep`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/api.py#L901-L916)

```python
sweep(
    path=""
)
```

경로를 분석하여 스윕을 반환합니다. 형식은 `entity/project/sweep_id`입니다.

| 인수 |  |
| :--- | :--- |
| `path` | (str, optional) 스윕의 경로로, 형식은 `entity/project/sweep_id`이어야 합니다. 만약 `api.entity`가 설정되어 있다면, `project/sweep_id` 형식을 사용할 수 있으며, `api.project`가 설정되어 있다면 sweep_id만 사용 가능합니다. |

| 반환값 |  |
| :--- | :--- |
| `Sweep` 오브젝트. |

### `sync_tensorboard`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/api.py#L436-L458)

```python
sync_tensorboard(
    root_dir, run_id=None, project=None, entity=None
)
```

tfevent 파일을 포함하는 로컬 디렉토리를 wandb와 동기화합니다.

### `team`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/api.py#L717-L726)

```python
team(
    team: str
) -> "public.Team"
```

주어진 이름과 일치하는 `Team`을 반환합니다.

| 인수 |  |
| :--- | :--- |
| `team` | (str) 팀의 이름입니다. |

| 반환값 |  |
| :--- | :--- |
| `Team` 오브젝트. |

### `user`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/api.py#L728-L748)

```python
user(
    username_or_email: str
) -> Optional['public.User']
```

사용자 이름 또는 이메일 주소로부터 user를 반환합니다.

참고: 이 함수는 로컬 관리자에게만 작동하며, 자신의 사용자 오브젝트를 가져오려면 `api.viewer`를 사용하세요.

| 인수 |  |
| :--- | :--- |
| `username_or_email` | (str) 사용자 이름 또는 이메일 주소 |

| 반환값 |  |
| :--- | :--- |
| `User` 오브젝트 또는 사용자를 찾을 수 없는 경우 None |

### `users`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/api.py#L750-L764)

```python
users(
    username_or_email: str
) -> List['public.User']
```

부분적인 사용자 이름 또는 이메일 주소를 통한 쿼리로 모든 사용자를 반환합니다.

참고: 이 함수는 로컬 관리자에게만 작동하며, 자신의 사용자 오브젝트를 가져오려면 `api.viewer`를 사용하세요.

| 인수 |  |
| :--- | :--- |
| `username_or_email` | (str) 찾고자 하는 사용자의 접두사 또는 접미사 |

| 반환값 |  |
| :--- | :--- |
| `User` 오브젝트 배열 |

| 클래스 변수 |  |
| :--- | :--- |
| `CREATE_PROJECT`<a id="CREATE_PROJECT"></a> |   |
| `DEFAULT_ENTITY_QUERY`<a id="DEFAULT_ENTITY_QUERY"></a> |   |
| `USERS_QUERY`<a id="USERS_QUERY"></a> |   |
| `VIEWER_QUERY`<a id="VIEWER_QUERY"></a> |   |