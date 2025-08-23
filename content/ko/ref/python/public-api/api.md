---
title: api
data_type_classification: module
menu:
  reference:
    identifier: ko-ref-python-public-api-api
object_type: public_apis_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/api.py >}}




# <kbd>module</kbd> `wandb.apis.public`
Public API를 사용하여 W&B에 저장된 데이터를 내보내거나 업데이트할 수 있습니다.

이 API를 사용하기 전에 스크립트에서 데이터를 먼저 기록해야 합니다. 자세한 내용은 [퀵스타트](https://docs.wandb.ai/quickstart)를 참고하세요.

Public API는 다음과 같은 상황에서 활용할 수 있습니다.
 - 완료된 실험의 메타데이터나 메트릭을 업데이트할 때
 - 결과를 데이터프레임으로 가져와 Jupyter 노트북에서 사후 분석을 할 때
 - `ready-to-deploy` 태그가 붙은 모델 artifact 들을 확인할 때

Public API 활용법에 대한 더 많은 정보는 [가이드](https://docs.wandb.com/guides/track/public-api-guide)를 참고하세요.


## <kbd>class</kbd> `Api`
W&B 서버에 쿼리를 보낼 때 사용합니다.



**예시:**
 ```python
import wandb

wandb.Api()
```

### <kbd>method</kbd> `Api.__init__`

```python
__init__(
    overrides: Optional[Dict[str, Any]] = None,
    timeout: Optional[int] = None,
    api_key: Optional[str] = None
) → None
```

API를 초기화합니다.



**파라미터:**
 
 - `overrides`:  W&B 서버가 기본 `https://api.wandb.ai` 가 아닐 경우, `base_url`을 설정할 수 있습니다. 또한 `entity`, `project`, `run`의 기본값을 설정할 수 있습니다.
 - `timeout`:  API 요청의 HTTP 타임아웃(초 단위)입니다. 지정하지 않으면 기본 설정이 사용됩니다.
 - `api_key`:  인증에 사용할 API 키입니다. 제공하지 않으면 현재 환경 또는 설정에서 API 키를 사용합니다.


---

### <kbd>property</kbd> Api.api_key

W&B API 키를 반환합니다.

---

### <kbd>property</kbd> Api.client

클라이언트 오브젝트를 반환합니다.

---

### <kbd>property</kbd> Api.default_entity

기본 W&B entity 를 반환합니다.

---

### <kbd>property</kbd> Api.user_agent

W&B public user agent를 반환합니다.

---

### <kbd>property</kbd> Api.viewer

viewer 오브젝트를 반환합니다.



---

### <kbd>method</kbd> `Api.artifact`

```python
artifact(name: str, type: Optional[str] = None)
```

단일 artifact 오브젝트를 반환합니다.



**파라미터:**
 
 - `name`:  artifact의 이름입니다. artifact의 이름은 최소 프로젝트명, artifact 이름, artifact의 버전 또는 에일리어스로 구성된 파일 경로와 유사합니다. entity를 앞에 `/`와 함께 추가할 수 있습니다. 이름에 entity가 명시되지 않으면 Run 또는 API 설정의 entity가 사용됩니다.
 - `type`:  가져올 artifact의 타입입니다.



**리턴값:**
 `Artifact` 오브젝트를 반환합니다.



**예외:**
 
 - `ValueError`:  artifact 이름이 지정되지 않은 경우
 - `ValueError`:  artifact type이 지정되었으나, 가져온 artifact의 타입과 일치하지 않는 경우



**예시:**
 아래 코드에서 "entity", "project", "artifact", "version", "alias"는 각각 W&B entity, artifact가 속한 프로젝트명, artifact 이름, artifact 버전을 나타내는 예시입니다.

```python
import wandb

# 프로젝트, artifact 이름, artifact 에일리어스 지정
wandb.Api().artifact(name="project/artifact:alias")

# 프로젝트, artifact 이름, 특정 artifact 버전 지정
wandb.Api().artifact(name="project/artifact:version")

# entity, 프로젝트, artifact 이름, artifact 에일리어스 지정
wandb.Api().artifact(name="entity/project/artifact:alias")

# entity, 프로젝트, artifact 이름, 특정 artifact 버전 지정
wandb.Api().artifact(name="entity/project/artifact:version")
```



**참고:**  
> 이 메서드는 외부에서 사용할 목적으로만 제공됩니다. wandb 저장소 코드 내에서는 `api.artifact()`를 호출하지 마세요.

---

### <kbd>method</kbd> `Api.artifact_collection`

```python
artifact_collection(type_name: str, name: str) → public.ArtifactCollection
```

타입별 artifact 컬렉션을 반환합니다.

반환된 `ArtifactCollection` 오브젝트를 사용해 해당 컬렉션의 개별 artifact 정보를 조회할 수 있습니다.



**파라미터:**
 
 - `type_name`:  가져올 artifact 컬렉션의 타입
 - `name`:  artifact 컬렉션의 이름. entity를 앞에 `/`와 함께 추가할 수 있습니다.



**리턴값:**
 `ArtifactCollection` 오브젝트를 반환합니다.



**예시:**
 아래 코드에서 "type", "entity", "project", "artifact_name"은 각각 컬렉션 타입, W&B entity, artifact가 속한 프로젝트명, artifact 이름입니다.

```python
import wandb

collections = wandb.Api().artifact_collection(
    type_name="type", name="entity/project/artifact_name"
)

# 컬렉션에서 첫 번째 artifact 가져오기
artifact_example = collections.artifacts()[0]

# artifact의 내용을 지정한 디렉토리에 다운로드
artifact_example.download()
```

---

### <kbd>method</kbd> `Api.artifact_collection_exists`

```python
artifact_collection_exists(name: str, type: str) → bool
```

특정 프로젝트와 entity 내에 artifact 컬렉션이 존재하는지 확인합니다.



**파라미터:**
 
 - `name`:  artifact 컬렉션 이름. entity를 앞에 `/`와 함께 추가할 수 있습니다. entity나 project가 명시되지 않으면 override 파라미터에서 추론합니다. 없으면 사용자 설정에서 entity를 가져오고, 프로젝트는 "uncategorized"로 기본 설정됩니다.
 - `type`:  artifact 컬렉션의 타입



**리턴값:**
 컬렉션이 존재하면 True, 아니면 False를 반환합니다.



**예시:**
 아래 코드에서 "type", "collection_name"은 artifact 컬렉션의 타입과 이름입니다.

```python
import wandb

wandb.Api.artifact_collection_exists(type="type", name="collection_name")
```

---

### <kbd>method</kbd> `Api.artifact_collections`

```python
artifact_collections(
    project_name: str,
    type_name: str,
    per_page: int = 50
) → public.ArtifactCollections
```

지정한 조건에 맞는 artifact 컬렉션 집합을 반환합니다.



**파라미터:**
 
 - `project_name`:  필터링할 프로젝트 이름
 - `type_name`:  필터링할 artifact 타입 이름
 - `per_page`:  페이지당 쿼리 사이즈를 설정합니다. None일 경우 기본 크기를 사용합니다. 일반적으로 변경할 필요는 없습니다.



**리턴값:**
 반복 가능한 `ArtifactCollections` 오브젝트를 반환합니다.

---

### <kbd>method</kbd> `Api.artifact_exists`

```python
artifact_exists(name: str, type: Optional[str] = None) → bool
```

지정한 프로젝트와 entity 내에 artifact 버전이 존재하는지 확인합니다.



**파라미터:**
 
 - `name`:  artifact의 이름. entity와 프로젝트를 앞에 붙여주며, artifact의 버전이나 에일리어스는 콜론으로 추가합니다. entity나 프로젝트가 지정되지 않으면, override 파라미터를 사용하며 없으면 설정에서 entity를, 프로젝트는 "Uncategorized"로 지정됩니다.
 - `type`:  artifact의 타입



**리턴값:**
 artifact 버전이 존재하면 True, 아니면 False를 반환합니다.



**예시:**
 아래에서 "entity", "project", "artifact", "version", "alias"는 각각 예시입니다.

```python
import wandb

wandb.Api().artifact_exists("entity/project/artifact:version")
wandb.Api().artifact_exists("entity/project/artifact:alias")
```

---

### <kbd>method</kbd> `Api.artifact_type`

```python
artifact_type(
    type_name: str,
    project: Optional[str] = None
) → public.ArtifactType
```

해당하는 `ArtifactType`을 반환합니다.



**파라미터:**
 
 - `type_name`:  조회할 artifact 타입 이름
 - `project`:  (선택) 필터로 사용할 프로젝트명 또는 경로



**리턴값:**
 `ArtifactType` 오브젝트를 반환합니다.

---

### <kbd>method</kbd> `Api.artifact_types`

```python
artifact_types(project: Optional[str] = None) → public.ArtifactTypes
```

조건에 맞는 artifact 타입들의 집합을 반환합니다.



**파라미터:**
 
 - `project`:  필터에 사용할 프로젝트명 또는 경로



**리턴값:**
 반복 가능한 `ArtifactTypes` 오브젝트를 반환합니다.

---

### <kbd>method</kbd> `Api.artifact_versions`

```python
artifact_versions(type_name, name, per_page=50)
```

Deprecated(더 이상 사용하지 않음). 대신 `Api.artifacts(type_name, name)` 메서드를 사용하세요.

---

### <kbd>method</kbd> `Api.artifacts`

```python
artifacts(
    type_name: str,
    name: str,
    per_page: int = 50,
    tags: Optional[List[str]] = None
) → public.Artifacts
```

`Artifacts` 컬렉션을 반환합니다.



**파라미터:**
 type_name: 가져올 artifact들의 타입  
 name: artifact 컬렉션명. entity를 앞에 `/`와 함께 추가할 수 있습니다.  
 per_page: 쿼리 페이지네이션을 위한 페이지 크기 설정. None이면 기본값 사용. 일반적으로 변경할 필요 없습니다.  
 tags: 이 태그들을 모두 가지고 있는 artifact만 반환합니다.



**리턴값:**
 반복 가능한 `Artifacts` 오브젝트를 반환합니다.



**예시:**
 예시 코드에서 "type", "entity", "project", "artifact_name"은 각각 artifact의 타입, W&B entity, artifact가 로깅된 프로젝트명, artifact 이름입니다.

```python
import wandb

wandb.Api().artifacts(type_name="type", name="entity/project/artifact_name")
```

---

### <kbd>method</kbd> `Api.automation`

```python
automation(name: str, entity: Optional[str] = None) → Automation
```

파라미터에 맞는 Automation 한 개를 반환합니다.



**파라미터:**
 
 - `name`:  가져올 automation의 이름
 - `entity`:  automation이 속한 entity



**예외:**
 
 - `ValueError`:  하나도 없거나 여러 개의 Automation이 검색될 경우



**예시:**
 "my-automation"이라는 automation 가져오기:

```python
import wandb

api = wandb.Api()
automation = api.automation(name="my-automation")
```

"my-team" entity 내 "other-automation" 가져오기:

```python
automation = api.automation(name="other-automation", entity="my-team")
```

---

### <kbd>method</kbd> `Api.automations`

```python
automations(
    entity: Optional[str] = None,
    name: Optional[str] = None,
    per_page: int = 50
) → Iterator[ForwardRef('Automation')]
```

주어진 파라미터에 맞는 모든 Automation을 반환하는 iterator입니다.

인자를 주지 않으면 현재 사용자가 엑세스할 수 있는 모든 Automation들이 iterator에 포함됩니다.



**파라미터:**
 
 - `entity`:  automation을 가져올 entity  
 - `name`:  automation의 이름
 - `per_page`:  페이지당 가져올 automation 개수 (기본 50, 일반적으로 변경할 필요 없음)



**리턴값:**
 automation 리스트를 반환합니다.



**예시:**
 "my-team" entity의 모든 automation 가져오기:

```python
import wandb

api = wandb.Api()
automations = api.automations(entity="my-team")
```

---

### <kbd>method</kbd> `Api.create_automation`

```python
create_automation(
    obj: 'NewAutomation',
    fetch_existing: bool = False,
    **kwargs: typing_extensions.Unpack[ForwardRef('WriteAutomationsKwargs')]
) → Automation
```

새로운 Automation을 만듭니다.



**파라미터:**
  obj:  생성할 automation  
  fetch_existing:  True인 경우, 같은 이름의 automation이 이미 존재하면 에러를 발생시키는 대신 기존 automation을 가져옵니다.  
  **kwargs:  automation 생성 전 할당할 추가 값들.  
        - `name`: automation 이름  
        - `description`: automation 설명  
        - `enabled`: 활성화 여부  
        - `scope`: automation의 scope  
        - `event`: automation을 트리거하는 이벤트  
        - `action`: 이벤트 발생 시 트리거되는 action    



**리턴값:**
  저장된 Automation 객체



**예시:**
 특정 프로젝트 내 run이 커스텀 threshold를 넘는 메트릭을 기록할 때 Slack 알림을 보내는 "my-automation" 만들기

```python
import wandb
from wandb.automations import OnRunMetric, RunEvent, SendNotification

api = wandb.Api()

project = api.project("my-project", entity="my-team")

# 팀의 첫 번째 Slack 인테그레이션 사용
slack_hook = next(api.slack_integrations(entity="my-team"))

event = OnRunMetric(
     scope=project,
     filter=RunEvent.metric("custom-metric") > 10,
)
action = SendNotification.from_integration(slack_hook)

automation = api.create_automation(
     event >> action,
     name="my-automation",
     description="custom-metric이 10을 초과하면 Slack 메시지를 전송.",
)
```

---

### <kbd>method</kbd> `Api.create_custom_chart`

```python
create_custom_chart(
    entity: str,
    name: str,
    display_name: str,
    spec_type: Literal['vega2'],
    access: Literal['private', 'public'],
    spec: Union[str, dict]
) → str
```

커스텀 차트 프리셋을 만들고, 그 id를 반환합니다.



**파라미터:**
 
 - `entity`:  차트를 소유한 사용자 또는 팀
 - `name`:  차트 프리셋의 고유 식별자
 - `display_name`:  UI에 표시될 사람 친화적 이름
 - `spec_type`:  명세 타입. Vega-Lite v2 명세만 "vega2"만 허용됩니다.
 - `access`:  차트 엑세스 레벨
        - "private": 차트를 만든 entity만 접근 가능
        - "public": 모든 사용자 공개
 - `spec`:  Vega 또는 Vega-Lite 명세(JSON string 또는 dict)



**리턴값:**
 생성된 차트 프리셋 id ("entity/name" 형식)



**예외:**
 
 - `wandb.Error`:  차트 생성 실패 시
 - `UnsupportedError`:  서버가 커스텀 차트를 지원하지 않을 경우



**예시:**
 ```python
    import wandb

    api = wandb.Api()

    # 간단한 바 차트 명세 정의
    vega_spec = {
         "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
         "mark": "bar",
         "data": {"name": "wandb"},
         "encoding": {
             "x": {"field": "${field:x}", "type": "ordinal"},
             "y": {"field": "${field:y}", "type": "quantitative"},
         },
    }

    # 커스텀 차트 생성
    chart_id = api.create_custom_chart(
         entity="my-team",
         name="my-bar-chart",
         display_name="나만의 커스텀 바차트",
         spec_type="vega2",
         access="private",
         spec=vega_spec,
    )

    # wandb.plot_table()에서 사용
    chart = wandb.plot_table(
         vega_spec_name=chart_id,
         data_table=my_table,
         fields={"x": "category", "y": "value"},
    )
    ```

---

### <kbd>method</kbd> `Api.create_project`

```python
create_project(name: str, entity: str) → None
```

새로운 프로젝트를 생성합니다.



**파라미터:**
 
 - `name`:  프로젝트 이름
 - `entity`:  프로젝트의 entity

---

### <kbd>method</kbd> `Api.create_registry`

```python
create_registry(
    name: str,
    visibility: Literal['organization', 'restricted'],
    organization: Optional[str] = None,
    description: Optional[str] = None,
    artifact_types: Optional[List[str]] = None
) → Registry
```

새로운 registry를 생성합니다.



**파라미터:**
 
 - `name`:  registry 이름. 조직 내에서 유일해야 함
 - `visibility`:  registry의 공개 정도
 - `organization`:  조직 내 모든 사용자가 이 registry를 볼 수 있습니다. 역할은 UI의 설정에서 나중에 변경할 수 있습니다.
 - `restricted`:  초대된 멤버만 UI를 통해 엑세스할 수 있고, 공개 공유는 비활성화됩니다.
 - `organization`:  registry의 조직명. 설정에 지정된 조직이 없으면, entity가 하나의 조직에만 속한 경우 그 조직에서 가져옵니다.
 - `description`:  registry 설명
 - `artifact_types`:  registry에서 허용할 artifact 타입들. 타입의 규칙: 128자 이하, `/` 또는 `:` 포함 불가. 지정하지 않으면, 모든 타입 가능. 허용된 타입은 나중에 삭제 불가.



**리턴값:**
 registry 오브젝트



**예시:**
 ```python
import wandb

api = wandb.Api()
registry = api.create_registry(
    name="my-registry",
    visibility="restricted",
    organization="my-org",
    description="테스트 registry입니다",
    artifact_types=["model"],
)
```

---

### <kbd>method</kbd> `Api.create_run`

```python
create_run(
    run_id: Optional[str] = None,
    project: Optional[str] = None,
    entity: Optional[str] = None
) → public.Run
```

새로운 run을 생성합니다.



**파라미터:**
 
 - `run_id`:  run에 할당할 ID. 지정하지 않으면 W&B에서 랜덤 ID를 생성합니다.
 - `project`:  run을 기록할 프로젝트. 지정하지 않으면 "Uncategorized"로 기록합니다.
 - `entity`:  프로젝트를 소유한 entity. 지정하지 않으면 기본 entity로 기록합니다.



**리턴값:**
 새로 생성된 `Run`을 반환합니다.

---

### <kbd>method</kbd> `Api.create_run_queue`

```python
create_run_queue(
    name: str,
    type: 'public.RunQueueResourceType',
    entity: Optional[str] = None,
    prioritization_mode: Optional[ForwardRef('public.RunQueuePrioritizationMode')] = None,
    config: Optional[dict] = None,
    template_variables: Optional[dict] = None
) → public.RunQueue
```

W&B Launch에서 새로운 run queue를 생성합니다.



**파라미터:**
 
 - `name`:  생성할 queue의 이름
 - `type`:  queue에서 사용할 리소스 타입. "local-container", "local-process", "kubernetes", "sagemaker", "gcp-vertex" 중 하나
 - `entity`:  queue를 생성할 entity명. `None`이면 기본 entity 사용
 - `prioritization_mode`:  우선순위 버전 ("V0" 또는 `None`)
 - `config`:  queue에서 사용할 기본 리소스 설정. 핸들바(예: `{{var}}`)로 템플릿 변수 지정 가능
 - `template_variables`:  config에 쓰일 템플릿 변수 스키마 사전



**리턴값:**
 새로 생성된 `RunQueue` 오브젝트



**예외:**
 파라미터가 잘못되었을 때 ValueError 발생, wandb API 에러 시 wandb.Error 발생

---

### <kbd>method</kbd> `Api.create_team`

```python
create_team(team: str, admin_username: Optional[str] = None) → public.Team
```

새로운 팀을 생성합니다.



**파라미터:**
 
 - `team`:  팀 이름
 - `admin_username`:  팀의 관리자 유저네임. 지정하지 않으면 현재 사용자가 관리자로 지정됩니다.



**리턴값:**
 `Team` 오브젝트 반환

---

### <kbd>method</kbd> `Api.create_user`

```python
create_user(email: str, admin: Optional[bool] = False)
```

새로운 사용자를 생성합니다.



**파라미터:**
 
 - `email`:  사용자 이메일 주소
 - `admin`:  사용자를 글로벌 인스턴스 어드민으로 지정할지 여부



**리턴값:**
 `User` 오브젝트 반환

---

### <kbd>method</kbd> `Api.delete_automation`

```python
delete_automation(obj: Union[ForwardRef('Automation'), str]) → Literal[True]
```

automation을 삭제합니다.



**파라미터:**
 
 - `obj`:  삭제할 automation 오브젝트 또는 해당 id



**리턴값:**
 삭제에 성공하면 True 반환

---

### <kbd>method</kbd> `Api.flush`

```python
flush()
```

로컬 캐시를 초기화합니다.

api 오브젝트는 run에 대한 로컬 캐시를 유지하므로, 스크립트 실행 중 run의 상태가 변경될 가능성이 있다면 `api.flush()`를 사용해 run 관련 최신 데이터를 가져올 수 있습니다.

---

### <kbd>method</kbd> `Api.from_path`

```python
from_path(path: str)
```

path를 이용해 run, sweep, project 또는 report를 반환합니다.



**파라미터:**
 
 - `path`:  project, run, sweep, report의 경로



**리턴값:**
 `Project`, `Run`, `Sweep`, 또는 `BetaReport` 인스턴스



**예외:**
 `wandb.Error`:  path가 올바르지 않거나 해당 오브젝트가 없는 경우



**예시:**
 "project", "team", "run_id", "sweep_id", "report_name"은 각각 예시값입니다.

```python
import wandb

api = wandb.Api()

project = api.from_path("project")
team_project = api.from_path("team/project")
run = api.from_path("team/project/runs/run_id")
sweep = api.from_path("team/project/sweeps/sweep_id")
report = api.from_path("team/project/reports/report_name")
```

---

### <kbd>method</kbd> `Api.integrations`

```python
integrations(
    entity: Optional[str] = None,
    per_page: int = 50
) → Iterator[ForwardRef('Integration')]
```

특정 entity의 모든 인테그레이션을 반환하는 iterator를 제공합니다.



**파라미터:**
 
 - `entity`:  인테그레이션을 조회할 entity(e.g. 팀명). 지정하지 않으면 사용자의 기본 entity가 사용됩니다.
 - `per_page`:  페이지당 반환할 인테그레이션 수 (기본 50, 대부분 변경할 필요 없음)



**Yield:**
 
 - `Iterator[SlackIntegration | WebhookIntegration]`:  지원되는 인테그레이션 iterator


---

### <kbd>method</kbd> `Api.job`

```python
job(name: Optional[str], path: Optional[str] = None) → public.Job
```

`Job` 오브젝트를 반환합니다.



**파라미터:**
 
 - `name`:  job의 이름
 - `path`:  job artifact를 다운로드할 루트 경로



**리턴값:**
 `Job` 오브젝트 반환

---

### <kbd>method</kbd> `Api.list_jobs`

```python
list_jobs(entity: str, project: str) → List[Dict[str, Any]]
```

지정한 entity와 프로젝트에 대해 jobs를 반환합니다.



**파라미터:**
 
 - `entity`:  job을 조회할 entity
 - `project`:  job을 조회할 프로젝트



**리턴값:**
 매칭되는 job들의 리스트를 반환

---

### <kbd>method</kbd> `Api.project`

```python
project(name: str, entity: Optional[str] = None) → public.Project
```

지정한 이름(및 entity, 옵션)에 해당하는 `Project` 반환



**파라미터:**
 
 - `name`:  프로젝트명
 - `entity`:  요청한 entity명. None이면 `Api`에 전달한 기본 entity를 사용함. 기본 entity 없으면 `ValueError` 발생



**리턴값:**
 `Project` 오브젝트 반환

---

### <kbd>method</kbd> `Api.projects`

```python
projects(entity: Optional[str] = None, per_page: int = 200) → public.Projects
```

특정 entity의 프로젝트 목록을 가져옵니다.



**파라미터:**
 
 - `entity`:  요청한 entity명. None이면 `Api`에 전달한 기본 entity 사용. 없으면 `ValueError` 발생
 - `per_page`:  페이지 사이즈 설정(None이면 기본값 사용). 거의 변경하지 않음



**리턴값:**
 `Project` 오브젝트들의 반복 가능한 컬렉션(`Projects`) 반환

---

### <kbd>method</kbd> `Api.queued_run`

```python
queued_run(
    entity: str,
    project: str,
    queue_name: str,
    run_queue_item_id: str,
    project_queue=None,
    priority=None
)
```

path를 기반으로 단일 queued run을 반환합니다.

`entity/project/queue_id/run_queue_item_id` 형식의 path 구문을 파싱합니다.

---

### <kbd>method</kbd> `Api.registries`

```python
registries(
    organization: Optional[str] = None,
    filter: Optional[Dict[str, Any]] = None
) → Registries
```

Registry iterator를 반환합니다.

이 iterator를 사용해 조직 내 registry, 컬렉션, artifact 버전을 검색/필터링할 수 있습니다.



**파라미터:**
 
 - `organization`:  (str, 옵션) 가져올 registry의 조직명. 지정하지 않으면 사용자 설정에 등록된 조직을 사용
 - `filter`:  (dict, 옵션) 각 registry object에 적용하는 MongoDB 스타일 filter.  
 컬렉션 필터 필드로는 `name`, `description`, `created_at`, `updated_at`  
 버전 필드로는 `tag`, `alias`, `created_at`, `updated_at`, `metadata` 사용 가능



**리턴값:**
 registry iterator



**예시:**
 "model"이 포함된 모든 registry 찾기

```python
import wandb

api = wandb.Api()  # entity가 여러 조직에 속해 있다면 org를 명시
api.registries(filter={"name": {"$regex": "model"}})
```

이름이 "my_collection"이고 태그가 "my_tag"인 컬렉션 찾기

```python
api.registries().collections(filter={"name": "my_collection", "tag": "my_tag"})
```

컬렉션 이름에 "my_collection"이 포함되고, alias가 "best"인 artifact 버전 찾기

```python
api.registries().collections(
    filter={"name": {"$regex": "my_collection"}}
).versions(filter={"alias": "best"})
```

"model"이 포함되고, tag가 "prod"이거나 alias가 "best"인 artifact 버전 찾기

```python
api.registries(filter={"name": {"$regex": "model"}}).versions(
    filter={"$or": [{"tag": "prod"}, {"alias": "best"}]}
)
```

---

### <kbd>method</kbd> `Api.registry`

```python
registry(name: str, organization: Optional[str] = None) → Registry
```

registry 이름을 기준으로 registry를 반환합니다.



**파라미터:**
 
 - `name`:  registry의 이름(`wandb-registry-` prefix 제외)
 - `organization`:  registry가 속한 조직명. 설정에 조직이 없으면 entity가 속한 조직에서 자동으로 결정



**리턴값:**
 registry 오브젝트



**예시:**
 registry를 불러오고 업데이트하기

```python
import wandb

api = wandb.Api()
registry = api.registry(name="my-registry", organization="my-org")
registry.description = "업데이트된 설명"
registry.save()
```

---

### <kbd>method</kbd> `Api.reports`

```python
reports(
    path: str = '',
    name: Optional[str] = None,
    per_page: int = 50
) → public.Reports
```

프로젝트 경로에 해당하는 report들을 조회합니다.

참고: `wandb.Api.reports()` API는 베타이며, 향후 릴리즈에서 변경될 수 있습니다.



**파라미터:**
 
 - `path`:  report가 속한 프로젝트 경로. 프로젝트를 만든 entity를 앞에 `/`와 함께 명시
 - `name`:  요청하는 report 이름
 - `per_page`:  페이지당 반환될 report 수. None이면 기본값 사용



**리턴값:**
 `BetaReport` 오브젝트의 반복 가능한 `Reports` 오브젝트



**예시:**
 ```python
import wandb

wandb.Api.reports("entity/project")
```

---

### <kbd>method</kbd> `Api.run`

```python
run(path='')
```

`entity/project/run_id` 형식의 path를 파싱하여 단일 run을 반환합니다.



**파라미터:**
 
 - `path`:  `entity/project/run_id` 형식의 run 경로. `api.entity`가 설정된 경우 `project/run_id`, `api.project`까지 있다면 run_id만 전달 가능



**리턴값:**
 `Run` 오브젝트 반환

---

### <kbd>method</kbd> `Api.run_queue`

```python
run_queue(entity: str, name: str)
```

해당 entity의 이름이 일치하는 `RunQueue`를 반환합니다.

run queue 생성 방법은 `Api.create_run_queue`를 참고하세요.

---

### <kbd>method</kbd> `Api.runs`

```python
runs(
    path: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    order: str = '+created_at',
    per_page: int = 50,
    include_sweeps: bool = True
)
```

필터 조건에 맞는 프로젝트 내 run 집합을 반환합니다.

필터링 가능한 필드는 다음과 같습니다:
- `createdAt`: run이 생성된 시각(ISO 8601 형식, 예: "2023-01-01T12:00:00Z")
- `displayName`: 사람이 읽기 쉬운 run 표시 이름(예: "eager-fox-1")
- `duration`: run의 전체 실행 시간(초)
- `group`: 연관된 run들을 묶는 그룹명
- `host`: run이 실행된 호스트명
- `jobType`: run의 job 유형 또는 목적
- `name`: run의 고유 식별자(예: "a1b2cdef")
- `state`: run의 현재 상태
- `tags`: run에 연결된 태그
- `username`: run을 시작한 사용자 이름

추가로, run config나 요약 메트릭(예: `config.experiment_name`, `summary_metrics.loss` 등)도 필터링할 수 있습니다.

보다 복잡한 필터링은 MongoDB 쿼리 오퍼레이터를 사용할 수 있습니다.  
자세한 내용: https://docs.mongodb.com/manual/reference/operator/query  
지원되는 연산:  
- `$and`  
- `$or`
- `$nor`
- `$eq`
- `$ne`
- `$gt`
- `$gte`
- `$lt`
- `$lte`
- `$in`
- `$nin`
- `$exists`
- `$regex`




**파라미터:**
 
 - `path`:  (str) 프로젝트 경로, 예: "entity/project"
 - `filters`:  (dict) MongoDB 쿼리 문법으로 run을 필터링. config.key, summary_metrics.key, state, entity, createdAt 등으로 필터링 가능
 - `For example`:  `{"config.experiment_name": "foo"}`로 experiment_name이 "foo"인 run 찾기
 - `order`:  (str) 정렬기준: `created_at`, `heartbeat_at`, `config.*.value`, `summary_metrics.*` 가능. +는 오름차순, -는 내림차순(기본). 기본값은 오래된 run부터 최신 run(`created_at` 오름차순)
 - `per_page`:  (int) 페이지 사이즈
 - `include_sweeps`:  (bool) sweep run도 결과에 포함할지 여부



**리턴값:**
 반복 가능한 `Run` 오브젝트 집합 (`Runs` 오브젝트)



**예시:**
 ```python
# 프로젝트에서 config.experiment_name이 "foo"인 run 찾기
api.runs(path="my_entity/project", filters={"config.experiment_name": "foo"})
```

```python
# 프로젝트에서 config.experiment_name이 "foo" 또는 "bar"인 run 찾기
api.runs(
    path="my_entity/project",
    filters={
         "$or": [
             {"config.experiment_name": "foo"},
             {"config.experiment_name": "bar"},
         ]
    },
)
```

```python
# regex로 config.experiment_name 필터링 (문자열 anchored match 미지원)
api.runs(
    path="my_entity/project",
    filters={"config.experiment_name": {"$regex": "b.*"}},
)
```

```python
# regex로 run 표시 이름 필터링 (문자열 anchored match 미지원)
api.runs(
    path="my_entity/project", filters={"display_name": {"$regex": "^foo.*"}}
)
```

```python
# 손실(loss) 오름차순 정렬
api.runs(path="my_entity/project", order="+summary_metrics.loss")
```

---

### <kbd>method</kbd> `Api.slack_integrations`

```python
slack_integrations(
    entity: Optional[str] = None,
    per_page: int = 50
) → Iterator[ForwardRef('SlackIntegration')]
```

특정 entity에 대한 Slack 인테그레이션 iterator를 반환합니다.



**파라미터:**
 
 - `entity`:  인테그레이션을 조회할 entity(e.g. 팀명). 지정하지 않으면 기본 entity 사용
 - `per_page`:  페이지당 인테그레이션 개수(기본 50, 일반적으로 변경 필요 없음)



**Yield:**
 
 - `Iterator[SlackIntegration]`:  Slack 인테그레이션 iterator



**예시:**
 "my-team"에 등록된 모든 Slack 인테그레이션 가져오기

```python
import wandb

api = wandb.Api()
slack_integrations = api.slack_integrations(entity="my-team")
```

채널 이름이 "team-alerts-"로 시작하는 Slack 인테그레이션만 필터링

```python
slack_integrations = api.slack_integrations(entity="my-team")
team_alert_integrations = [
    ig
    for ig in slack_integrations
    if ig.channel_name.startswith("team-alerts-")
]
```

---

### <kbd>method</kbd> `Api.sweep`

```python
sweep(path='')
```

`entity/project/sweep_id` 형식의 path를 파싱하여 sweep을 반환합니다.



**파라미터:**
 
 - `path`:  `entity/project/sweep_id` 형식의 sweep 경로. `api.entity`가 있을 경우 `project/sweep_id`, `api.project`가 있으면 sweep_id만 전달 가능



**리턴값:**
 `Sweep` 오브젝트 반환

---

### <kbd>method</kbd> `Api.sync_tensorboard`

```python
sync_tensorboard(root_dir, run_id=None, project=None, entity=None)
```

tfevent 파일이 포함된 로컬 디렉토리를 wandb와 동기화합니다.

---

### <kbd>method</kbd> `Api.team`

```python
team(team: str) → public.Team
```

주어진 이름의 `Team`을 반환합니다.



**파라미터:**
 
 - `team`:  팀 이름



**리턴값:**
 `Team` 오브젝트 반환

---

### <kbd>method</kbd> `Api.update_automation`

```python
update_automation(
    obj: 'Automation',
    create_missing: bool = False,
    **kwargs: typing_extensions.Unpack[ForwardRef('WriteAutomationsKwargs')]
) → Automation
```

기존 automation을 업데이트합니다.



**파라미터:**
 
 - `obj`:  업데이트할 automation(이미 존재하는 automation이어야 함)
 - create_missing (bool):  True면 automation이 존재하지 않을 경우 새로 생성
 - **kwargs:  업데이트 전에 할당할 추가 값들. 기존 속성을 덮어쓸 수도 있음
        - `name`: automation 이름
        - `description`: 설명
        - `enabled`: 활성화 여부
        - `scope`: 범위
        - `event`: 트리거 이벤트
        - `action`: 트리거 후 실행되는 액션



**리턴값:**
 업데이트된 automation 반환



**예시:**
 기존 automation("my-automation") 비활성화 및 설명 수정:

```python
import wandb

api = wandb.Api()

automation = api.automation(name="my-automation")
automation.enabled = False
automation.description = "참고용, 더 이상 사용하지 않음."

updated_automation = api.update_automation(automation)
```

또는

```python
import wandb

api = wandb.Api()

automation = api.automation(name="my-automation")

updated_automation = api.update_automation(
    automation,
    enabled=False,
    description="참고용, 더 이상 사용하지 않음.",
)
```

---

### <kbd>method</kbd> `Api.upsert_run_queue`

```python
upsert_run_queue(
    name: str,
    resource_config: dict,
    resource_type: 'public.RunQueueResourceType',
    entity: Optional[str] = None,
    template_variables: Optional[dict] = None,
    external_links: Optional[dict] = None,
    prioritization_mode: Optional[ForwardRef('public.RunQueuePrioritizationMode')] = None
)
```

W&B Launch에서 run queue upsert(없으면 생성, 있으면 변경)합니다.



**파라미터:**
 
 - `name`:  생성할 queue 이름
 - `entity`:  queue를 생성할 entity명. 없으면 기본 entity 사용
 - `resource_config`:  queue에 사용할 기본 리소스 설정. 핸들바(예: `{{var}}`)로 템플릿 변수 지정
 - `resource_type`:  사용할 리소스 타입. "local-container", "local-process", "kubernetes", "sagemaker", "gcp-vertex" 중 하나
 - `template_variables`:  config에 사용할 템플릿 변수 스키마 딕셔너리
 - `external_links`:  queue에서 사용할 외부 링크 dict(선택)
 - `prioritization_mode`:  우선순위 버전("V0" 또는 None)



**리턴값:**
 upsert된 `RunQueue` 오브젝트 반환



**예외:**
 파라미터가 잘못되었을 때 ValueError, wandb API 에러 시 wandb.Error

---

### <kbd>method</kbd> `Api.user`

```python
user(username_or_email: str) → Optional[ForwardRef('public.User')]
```

username 혹은 이메일 주소로 사용자를 반환합니다.

로컬 관리자만 사용 가능합니다. 본인 user 오브젝트는 `api.viewer`를 사용하세요.



**파라미터:**
 
 - `username_or_email`:  사용자 이름 또는 이메일 주소



**리턴값:**
 `User` 오브젝트 또는 사용자가 없을 경우 None

---

### <kbd>method</kbd> `Api.users`

```python
users(username_or_email: str) → List[ForwardRef('public.User')]
```

username 혹은 이메일 주소 일부로 모든 사용자를 반환합니다.

로컬 관리자만 사용 가능합니다. 본인 user 오브젝트는 `api.viewer`를 사용하세요.



**파라미터:**
 
 - `username_or_email`:  찾고자 하는 사용자의 접두/접미어



**리턴값:**
 `User` 오브젝트의 리스트

---

### <kbd>method</kbd> `Api.webhook_integrations`

```python
webhook_integrations(
    entity: Optional[str] = None,
    per_page: int = 50
) → Iterator[ForwardRef('WebhookIntegration')]
```

특정 entity의 webhook 인테그레이션 iterator를 반환합니다.



**파라미터:**
 
 - `entity`:  인테그레이션을 조회할 entity(e.g. 팀명). 없으면 기본 entity 사용
 - `per_page`:  페이지당 인테그레이션 개수(기본값 50, 일반적으로 변경 불필요)



**Yield:**
 
 - `Iterator[WebhookIntegration]`:  webhook 인테그레이션 iterator



**예시:**
 "my-team"의 모든 webhook 인테그레이션 조회

```python
import wandb

api = wandb.Api()
webhook_integrations = api.webhook_integrations(entity="my-team")
```

"my-fake-url.com"에 POST하는 webhook 인테그레이션만 필터링

```python
webhook_integrations = api.webhook_integrations(entity="my-team")
my_webhooks = [
    ig
    for ig in webhook_integrations
    if ig.url_endpoint.startswith("https://my-fake-url.com")
]
```