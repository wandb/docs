
# Api

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/api.py#L101-L1047' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>


wandb 서버를 조회하는 데 사용됩니다.

```python
Api(
    overrides=None,
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
|  `overrides` |  (dict) https://api.wandb.ai가 아닌 다른 wandb 서버를 사용하는 경우 `base_url`을 설정할 수 있습니다. 또한 `entity`, `project`, `run`에 대한 기본값을 설정할 수도 있습니다. |

| 속성 |  |
| :--- | :--- |

## 메서드

### `artifact`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/api.py#L944-L968)

```python
artifact(
    name, type=None
)
```

`entity/project/name` 형식의 경로를 파싱하여 단일 아티팩트를 반환합니다.

| 인수 |  |
| :--- | :--- |
|  `name` |  (str) 아티팩트 이름입니다. entity/project로 접두사가 붙을 수 있습니다. 유효한 이름은 다음 형식일 수 있습니다: name:version name:alias |
|  `type` |  (str, 선택사항) 가져올 아티팩트의 유형입니다. |

| 반환값 |  |
| :--- | :--- |
|  `Artifact` 개체입니다. |

### `artifact_collection`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/api.py#L913-L927)

```python
artifact_collection(
    type_name: str,
    name: str
)
```

유형 및 `entity/project/name` 형식의 경로를 파싱하여 단일 아티팩트 컬렉션을 반환합니다.

| 인수 |  |
| :--- | :--- |
|  `name` |  (str) 아티팩트 컬렉션 이름입니다. entity/project로 접두사가 붙을 수 있습니다. |
|  `type` |  (str) 가져올 아티팩트 컬렉션의 유형입니다. |

| 반환값 |  |
| :--- | :--- |
|  `ArtifactCollection` 개체입니다. |

### `artifact_collections`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/api.py#L906-L911)

```python
artifact_collections(
    project_name: str,
    type_name: str,
    per_page=50
)
```

### `artifact_type`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/api.py#L901-L904)

```python
artifact_type(
    type_name, project=None
)
```

### `artifact_types`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/api.py#L896-L899)

```python
artifact_types(
    project=None
)
```

### `artifact_versions`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/api.py#L929-L935)

```python
artifact_versions(
    type_name, name, per_page=50
)
```

사용 중단됨. 대신 artifacts(type_name, name)을 사용하세요.

### `artifacts`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/api.py#L937-L942)

```python
artifacts(
    type_name, name, per_page=50
)
```

### `create_project`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/api.py#L276-L277)

```python
create_project(
    name: str,
    entity: str
)
```

### `create_report`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/api.py#L285-L300)

```python
create_report(
    project: str,
    entity: str = "",
    title: Optional[str] = "제목 없는 리포트",
    description: Optional[str] = "",
    width: Optional[str] = "readable",
    blocks: Optional['wandb.apis.reports.util.Block'] = None
) -> "wandb.apis.reports.Report"
```

### `create_run`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/api.py#L279-L283)

```python
create_run(
    **kwargs
)
```

새로운 실행을 생성합니다.

### `create_run_queue`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/api.py#L302-L412)

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

새로운 실행 대기열(런치)을 생성합니다.

| 인수 |  |
| :--- | :--- |
|  `name` |  (str) 생성할 대기열의 이름입니다 |
|  `type` |  (str) 대기열에 사용될 리소스의 유형입니다. "local-container", "local-process", "kubernetes", "sagemaker", 또는 "gcp-vertex" 중 하나입니다. |
|  `entity` |  (str) 대기열을 생성할 엔티티의 이름입니다. None인 경우, 구성된 또는 기본 엔티티를 사용합니다. |
|  `prioritization_mode` |  (str) 사용할 우선 순위 결정 버전입니다. "V0" 또는 None입니다 |
|  `config` |  (dict) 대기열에 사용할 기본 리소스 구성입니다. 템플릿 변수를 지정하려면 중괄호(예: "{{var}}")를 사용합니다. template_variables (dict): 구성에 사용될 템플릿 변수 스키마의 사전입니다. 예상 형식: { "var-name": { "schema": { "type": "<string | number | integer>", "default": <선택적 값>, "minimum": <선택적 최소값>, "maximum": <선택적 최대값>, "enum": [..."<옵션>"] } } } |

| 반환값 |  |
| :--- | :--- |
|  새로 생성된 `RunQueue`입니다. |

| 예외 |  |
| :--- | :--- |
|  파라미터가 유효하지 않은 경우 ValueError wandb API 오류 시 wandb.Error |

### `create_team`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/api.py#L693-L703)

```python
create_team(
    team, admin_username=None
)
```

새로운 팀을 생성합니다.

| 인수 |  |
| :--- | :--- |
|  `team` |  (str) 팀의 이름입니다 |
|  `admin_username` |  (str) 팀의 관리자 사용자의 사용자 이름입니다. 기본값은 현재 사용자입니다. |

| 반환값 |  |
| :--- | :--- |
|  `Team` 개체입니다 |

### `create_user`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/api.py#L430-L440)

```python
create_user(
    email, admin=(False)
)
```

새로운 사용자를 생성합니다.

| 인수 |  |
| :--- | :--- |
|  `email` |  (str) 사용자의 이메일 주소입니다 |
|  `admin` |  (bool) 이 사용자가 전역 인스턴스 관리자여야 하는지 여부입니다 |

| 반환값 |  |
| :--- | :--- |
|  `User` 개체입니다 |

### `flush`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/api.py#L507-L514)

```python
flush()
```

로컬 캐시를 비웁니다.

api 개체는 실행의 로컬 캐시를 유지하므로 스크립트를 실행하는 동안 실행의 상태가 변경될 수 있으므로 `api.flush()`로 로컬 캐시를 지워 실행과 관련된 최신 값을 얻어야 합니다.

### `from_path`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/api.py#L516-L570)

```python
from_path(
    path
)
```

경로에서 실행, 스윕, 프로젝트 또는 리포트를 반환합니다.

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
|  `path` |  (str) 프로젝트, 실행, 스윕 또는 리포트의 경로입니다 |

| 반환값 |  |
| :--- | :--- |
|  `Project`, `Run`, `Sweep`, 또는 `BetaReport` 인스턴스입니다. |

| 예외 |  |
| :--- | :--- |
|  경로가 유효하지 않거나 개체가 존재하지 않는 경우 wandb.Error |

### `job`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/api.py#L970-L978)

```python
job(
    name, path=None
)
```

### `list_jobs`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/api.py#L980-L1047)

```python
list_jobs(
    entity, project
)
```

### `load_report`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/api.py#L414-L428)

```python
load_report(
    path: str
) -> "wandb.apis.reports.Report"
```

주어진 경로의 리포트를 가져옵니다.

| 인수 |  |
| :--- | :--- |
|  `path` |  (str) 대상 리포트의 경로로, `entity/project/reports/reportId` 형식입니다. wandb URL 뒤에 있는 URL을 복사하여 붙여넣기하여 이를 얻을 수 있습니다. 예: `megatruong/report-editing/reports/My-fabulous-report-title--VmlldzoxOTc1Njk0` |

| 반환값 |  |
| :--- | :--- |
|  `path`에서의 리포트를 나타내는 `BetaReport` 개체입니다. |

| 예외 |  |
| :--- | :--- |
|  경로가 유효하지 않은 경우 wandb.Error |

### `project`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/api.py#L657-L660)

```python
project(
    name, entity=None
)
```

### `projects`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/api.py#L632-L655)

```python
projects(
    entity=None, per_page=200
)
```

주어진 엔티티에 대한 프로젝트를 가져옵니다.

| 인수 |  |
| :--- | :--- |
|  `entity` |  (str) 요청된 엔티티의 이름입니다. None인 경우, `Api`에 전달된 기본 엔티티로 대체됩니다. 기본 엔티티가 없는 경우 `ValueError`를 발생시킵니다. |
|  `per_page` |  (int) 쿼리 페이징을 위한 페이지 크기를 설정합니다. None은 기본 크기를 사용합니다. 보통 이를 변경할 필요는 없습니다. |

| 반환값 |  |
| :--- | :--- |
|  `Project` 개체의 반복 가능한 컬렉션인 `Projects` 개체입니다. |

### `queued_run`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/api.py#L841-L862)

```python
queued_run(
    entity, project, queue_name, run_queue_item_id, project_queue=None,
    priority=None
)
```

경로를 파싱하여 단일 대기 중인 실행을 반환합니다.

`entity/project/queue_id/run_queue_item_id` 형식의 경로를 파싱합니다.

### `reports`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/api.py#L662-L691)

```python
reports(
    path="", name=None, per_page=50
)
```

주어진 프로젝트 경로에 대한 리포트를 가져옵니다.

경고: 이 API는 베타 버전이며 향후 릴리스에서 변경될 가능성이 높습니다

| 인수 |  |
| :--- | :--- |
|  `path` |  (str) 리포트가 위치한 프로젝트의 경로로, "entity/project" 형식이어야 합니다. |
|  `name` |  (str) 요청된 리포트의 이름입니다. |
|  `per_page` |  (int) 쿼리 페이징을 위한 페이지 크기를 설정합니다. None은 기본 크기를 사용합니다. 보통 이를 변경할 필요는

#### 예시:

"foo"로 설정된 config.experiment_name을 가진 my_project의 실행 찾기

```
api.runs(path="my_entity/my_project", filters={"config.experiment_name": "foo"})
```

"foo" 또는 "bar"로 설정된 config.experiment_name을 가진 my_project의 실행 찾기

```
api.runs(
    path="my_entity/my_project",
    filters={"$or": [{"config.experiment_name": "foo"}, {"config.experiment_name": "bar"}]}
)
```

정규 표현식과 일치하는 config.experiment_name을 가진 my_project의 실행 찾기 (앵커는 지원되지 않음)

```
api.runs(
    path="my_entity/my_project",
    filters={"config.experiment_name": {"$regex": "b.*"}}
)
```

정규 표현식과 일치하는 실행 이름을 가진 my_project의 실행 찾기 (앵커는 지원되지 않음)

```
api.runs(
    path="my_entity/my_project",
    filters={"display_name": {"$regex": "^foo.*"}}
)
```

손실이 오름차순으로 정렬된 my_project의 실행 찾기

```
api.runs(path="my_entity/my_project", order="+summary_metrics.loss")
```

| 인수 |  |
| :--- | :--- |
|  `path` |  (str) 프로젝트까지의 경로, "entity/project" 형태여야 함 |
|  `filters` |  (dict) MongoDB 쿼리 언어를 사용하여 특정 실행을 쿼리합니다. config.key, summary_metrics.key, state, entity, createdAt 등의 실행 속성으로 필터링할 수 있습니다. 예를 들어: {"config.experiment_name": "foo"}는 실험 이름이 "foo"로 설정된 실행을 찾습니다. 더 복잡한 쿼리를 만들기 위해 연산을 구성할 수 있으며, 언어에 대한 참조는 https://docs.mongodb.com/manual/reference/operator/query 에서 확인할 수 있습니다. |
|  `order` |  (str) 정렬 순서는 `created_at`, `heartbeat_at`, `config.*.value`, 또는 `summary_metrics.*`일 수 있습니다. 정렬 앞에 +를 붙이면 오름차순입니다. 정렬 앞에 -를 붙이면 내림차순입니다 (기본값). 기본 정렬 순서는 가장 최신에서 가장 오래된 순으로 run.created_at입니다. |

| 반환값 |  |
| :--- | :--- |
|  `Run` 객체의 반복 가능한 컬렉션인 `Runs` 객체. |

### `sweep`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/api.py#L879-L894)

```python
sweep(
    path=""
)
```

`entity/project/sweep_id` 형태의 경로를 파싱하여 스윕을 반환합니다.

| 인수 |  |
| :--- | :--- |
|  `path` |  (str, 선택사항) entity/project/sweep_id 형태의 스윕 경로입니다. `api.entity`가 설정된 경우 project/sweep_id 형태가 될 수 있으며, `api.project`가 설정된 경우 스윕_id만 될 수 있습니다. |

| 반환값 |  |
| :--- | :--- |
|  `Sweep` 객체. |

### `sync_tensorboard`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/api.py#L442-L464)

```python
sync_tensorboard(
    root_dir, run_id=None, project=None, entity=None
)
```

tfevent 파일을 포함하는 로컬 디렉터리를 wandb에 동기화합니다.

### `team`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/api.py#L705-L706)

```python
team(
    team
)
```

### `user`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/api.py#L708-L728)

```python
user(
    username_or_email
)
```

사용자 이름 또는 이메일 주소로 사용자를 반환합니다.

참고: 이 함수는 로컬 관리자에게만 작동합니다. 자신의 사용자 개체를 얻으려면 `api.viewer`를 사용하세요.

| 인수 |  |
| :--- | :--- |
|  `username_or_email` |  (str) 사용자의 사용자 이름 또는 이메일 주소 |

| 반환값 |  |
| :--- | :--- |
|  사용자를 찾을 수 없으면 None이, 그렇지 않으면 `User` 객체 |

### `users`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/api.py#L730-L744)

```python
users(
    username_or_email
)
```

부분적인 사용자 이름 또는 이메일 주소 쿼리로 모든 사용자를 반환합니다.

참고: 이 함수는 로컬 관리자에게만 작동합니다. 자신의 사용자 개체를 얻으려면 `api.viewer`를 사용하세요.

| 인수 |  |
| :--- | :--- |
|  `username_or_email` |  (str) 찾고자 하는 사용자의 접두사 또는 접미사 |

| 반환값 |  |
| :--- | :--- |
|  `User` 객체의 배열 |

| 클래스 변수 |  |
| :--- | :--- |
|  `CREATE_PROJECT`<a id="CREATE_PROJECT"></a> |   |
|  `USERS_QUERY`<a id="USERS_QUERY"></a> |   |
|  `VIEWER_QUERY`<a id="VIEWER_QUERY"></a> |   |