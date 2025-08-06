---
title: run
data_type_classification: module
menu:
  reference:
    identifier: ko-ref-python-public-api-runs
object_type: public_apis_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/runs.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API for Runs.

이 모듈은 W&B run 및 관련 데이터와 상호작용할 수 있는 클래스를 제공합니다.



**예시:**
 ```python
from wandb.apis.public import Api

# 필터에 맞는 run 가져오기
runs = Api().runs(
     path="entity/project", filters={"state": "finished", "config.batch_size": 32}
)

# run 데이터 엑세스
for run in runs:
     print(f"Run: {run.name}")
     print(f"Config: {run.config}")
     print(f"Metrics: {run.summary}")

     # pandas로 history 가져오기
     history_df = run.history(keys=["loss", "accuracy"], pandas=True)

     # Artifacts 사용하기
     for artifact in run.logged_artifacts():
         print(f"Artifact: {artifact.name}")
```



**안내:**

> 이 모듈은 W&B Public API의 일부로, run 데이터에 대한 읽기/쓰기 엑세스를 제공합니다. 새로운 run을 로그하려면 메인 wandb 패키지의 wandb.init() 함수를 사용하세요.

## <kbd>class</kbd> `Runs`
프로젝트와 선택적 필터에 연결된 run의 반복 가능한 컬렉션입니다.

일반적으로 `Api.runs` 네임스페이스를 통해 간접적으로 사용됩니다.



**인수:**
 
 - `client`:  (`wandb.apis.public.RetryingClient`) 요청에 사용할 API 클라이언트입니다.
 - `entity`:  (str) 프로젝트를 소유한 entity (사용자명 또는 팀) 입니다.
 - `project`:  (str) run을 가져올 프로젝트 이름입니다.
 - `filters`:  (Optional[Dict[str, Any]]) run 쿼리에 적용할 필터의 사전입니다.
 - `order`:  (Optional[str]) run의 정렬 순서로, "asc" 또는 "desc" 중 선택. 기본값은 "desc"입니다.
 - `per_page`:  (int) 요청당 가져올 run의 수 (기본값 50).
 - `include_sweeps`:  (bool) run에 sweep 정보를 포함할지 여부. 기본값 True.



**예시:**
 ```python
from wandb.apis.public.runs import Runs
from wandb.apis.public import Api

# 필터를 만족하는 프로젝트의 모든 run 가져오기
filters = {"state": "finished", "config.optimizer": "adam"}

runs = Api().runs(
    client=api.client,
    entity="entity",
    project="project_name",
    filters=filters,
)

# runs를 순회하며 세부 정보 출력
for run in runs:
    print(f"Run name: {run.name}")
    print(f"Run ID: {run.id}")
    print(f"Run URL: {run.url}")
    print(f"Run state: {run.state}")
    print(f"Run config: {run.config}")
    print(f"Run summary: {run.summary}")
    print(f"Run history (samples=5): {run.history(samples=5)}")
    print("----------")

# 특정 메트릭에 대한 모든 run의 histories 가져오기
histories_df = runs.histories(
    samples=100,  # run마다 샘플 개수
    keys=["loss", "accuracy"],  # 가져올 메트릭
    x_axis="_step",  # x축 메트릭
    format="pandas",  # pandas DataFrame 으로 반환
)
```

### <kbd>method</kbd> `Runs.__init__`

```python
__init__(
    client: 'RetryingClient',
    entity: str,
    project: str,
    filters: Optional[Dict[str, Any]] = None,
    order: Optional[str] = None,
    per_page: int = 50,
    include_sweeps: bool = True
)
```






---


### <kbd>property</kbd> Runs.length





---



### <kbd>method</kbd> `Runs.histories`

```python
histories(
    samples: int = 500,
    keys: Optional[List[str]] = None,
    x_axis: str = '_step',
    format: Literal['default', 'pandas', 'polars'] = 'default',
    stream: Literal['default', 'system'] = 'default'
)
```

필터 조건에 맞는 모든 run에 대해 샘플링된 히스토리 메트릭을 반환합니다.



**인수:**
 
 - `samples`:  run별 반환 샘플 수
 - `keys`:  특정 키의 메트릭만 반환
 - `x_axis`:  해당 메트릭을 x축으로 사용, 기본은 _step
 - `format`:  반환 데이터 형식, "default", "pandas", "polars" 중 선택
 - `stream`:  "default"는 일반 메트릭, "system"은 시스템 메트릭

**반환값:**
 
 - `pandas.DataFrame`:  `format="pandas"`일 때 history 메트릭의 `pandas.DataFrame` 반환
 - `polars.DataFrame`:  `format="polars"`일 때 history 메트릭의 `polars.DataFrame` 반환
 - `list of dicts`:  `format="default"`일 때 메트릭이 담긴 dict 리스트와 `run_id` key 반환


---

## <kbd>class</kbd> `Run`
특정 entity 및 프로젝트에 연결된 단일 run 입니다.



**인수:**
 
 - `client`:  W&B API 클라이언트
 - `entity`:  이 run과 연결된 entity
 - `project`:  이 run과 연결된 프로젝트
 - `run_id`:  run에 대한 고유 식별자
 - `attrs`:  run의 속성
 - `include_sweeps`:  run에 sweep 정보를 포함할지 여부



**속성:**
 
 - `tags` ([str]):  run에 연결된 태그 목록
 - `url` (str):  해당 run의 url
 - `id` (str):  run의 고유 식별자 (기본 8자)
 - `name` (str):  run 이름
 - `state` (str):  상태 중 하나: running, finished, crashed, killed, preempting, preempted
 - `config` (dict):  run에 연결된 하이퍼파라미터 사전
 - `created_at` (str):  run이 시작된 ISO 타임스탬프
 - `system_metrics` (dict):  run에 대해 기록된 최신 시스템 메트릭
 - `summary` (dict):  현재 summary를 가지는 mutable 딕셔너리 속성. update 호출 시 변경사항이 저장됨
 - `project` (str):  run이 소속된 프로젝트
 - `entity` (str):  run이 소속된 entity 이름
 - `project_internal_id` (int):  프로젝트의 내부 id
 - `user` (str):  run을 생성한 사용자 이름
 - `path` (str):  고유 식별자 [entity]/[project]/[run_id]
 - `notes` (str):  run에 대한 노트
 - `read_only` (boolean):  해당 run이 편집 가능한지 여부
 - `history_keys` (str):  로그된 history 메트릭의 키값 목록
 - `with `wandb.log({key`:  value})`
 - `metadata` (str):  wandb-metadata.json에서 가져온 run에 대한 메타데이터

### <kbd>method</kbd> `Run.__init__`

```python
__init__(
    client: 'RetryingClient',
    entity: str,
    project: str,
    run_id: str,
    attrs: Optional[Mapping] = None,
    include_sweeps: bool = True
)
```

Run 오브젝트를 초기화합니다.

Run은 항상 wandb.Api 인스턴스의 api.runs() 호출을 통해 초기화됩니다.


---

### <kbd>property</kbd> Run.entity

이 run과 연결된 entity 입니다.

---

### <kbd>property</kbd> Run.id

해당 run의 고유 식별자입니다.

---


### <kbd>property</kbd> Run.lastHistoryStep

run의 히스토리에서 마지막 step을 반환합니다.

---

### <kbd>property</kbd> Run.metadata

wandb-metadata.json에서 가져온 run에 대한 메타데이터입니다.

메타데이터에는 run 설명, 태그, 시작 시간, 메모리 사용량 등이 포함됩니다.

---

### <kbd>property</kbd> Run.name

run의 이름입니다.

---

### <kbd>property</kbd> Run.path

run의 path 정보입니다. path는 entity, project, run_id로 구성된 리스트입니다.

---

### <kbd>property</kbd> Run.state

해당 run의 상태. Finished, Failed, Crashed, Running 중 하나의 값이 올 수 있습니다.

---

### <kbd>property</kbd> Run.storage_id

run의 고유 storage 식별자입니다.

---

### <kbd>property</kbd> Run.summary

run에 연결된 summary 값을 담는 mutable dict-like 속성입니다.

---

### <kbd>property</kbd> Run.url

run의 URL입니다.

run URL은 entity, project, run_id로 생성됩니다. SaaS 사용자의 경우, `https://wandb.ai/entity/project/run_id` 형태를 가집니다.

---

### <kbd>property</kbd> Run.username

이 API는 deprecated 되었습니다. 대신 `entity`를 사용하세요.



---

### <kbd>classmethod</kbd> `Run.create`

```python
create(
    api,
    run_id=None,
    project=None,
    entity=None,
    state: Literal['running', 'pending'] = 'running'
)
```

주어진 프로젝트에 run을 생성합니다.

---

### <kbd>method</kbd> `Run.delete`

```python
delete(delete_artifacts=False)
```

해당 run을 wandb 백엔드에서 삭제합니다.



**인수:**
 
 - `delete_artifacts` (bool, optional):  run에 연결된 artifacts까지 삭제할지 여부

---

### <kbd>method</kbd> `Run.file`

```python
file(name)
```

artifact에서 주어진 이름의 파일 path를 반환합니다.



**인수:**
 
 - `name` (str):  요청할 파일의 이름



**반환값:**
 name 인수와 일치하는 `File` 반환

---

### <kbd>method</kbd> `Run.files`

```python
files(names=None, per_page=50)
```

지정한 파일의 경로를 각각 반환합니다.



**인수:**
 
 - `names` (list):  요청하는 파일명들, 비어 있으면 모든 파일 반환
 - `per_page` (int):  페이지당 결과 개수



**반환값:**
 `File` 오브젝트 위를 순회할 수 있는 `Files` 오브젝트 반환

---

### <kbd>method</kbd> `Run.history`

```python
history(samples=500, keys=None, x_axis='_step', pandas=True, stream='default')
```

해당 run에 대해 샘플링된 history 메트릭을 반환합니다.

history 레코드가 샘플링된 상태라도 괜찮다면, 이 방법이 가장 쉽고 빠릅니다.



**인수:**
 
 - `samples `:  (int, optional) 반환할 샘플 개수
 - `pandas `:  (bool, optional) pandas DataFrame 형태로 반환
 - `keys `:  (list, optional) 특정 키의 메트릭만 반환
 - `x_axis `:  (str, optional) 이 메트릭을 x축으로 사용, 기본값 _step
 - `stream `:  (str, optional) "default"는 일반 메트릭, "system"은 시스템 메트릭



**반환값:**
 
 - `pandas.DataFrame`:  pandas=True일 경우 history 메트릭의 `pandas.DataFrame` 반환
 - `list of dicts`:  pandas=False일 경우 history 메트릭의 dict 리스트 반환

---

### <kbd>method</kbd> `Run.load`

```python
load(force=False)
```





---

### <kbd>method</kbd> `Run.log_artifact`

```python
log_artifact(
    artifact: 'wandb.Artifact',
    aliases: Optional[Collection[str]] = None,
    tags: Optional[Collection[str]] = None
)
```

run의 output으로 artifact를 선언합니다.



**인수:**
 
 - `artifact` (`Artifact`):  `wandb.Api().artifact(name)`에서 반환된 artifact
 - `aliases` (list, optional):  이 artifact에 적용할 에일리어스
 - `tags`:  (list, optional) 이 artifact에 적용할 태그



**반환값:**
 `Artifact` 오브젝트 반환

---

### <kbd>method</kbd> `Run.logged_artifacts`

```python
logged_artifacts(per_page: int = 100) → RunArtifacts
```

이 run에서 기록된 모든 artifact를 가져옵니다.

run 동안 기록된 출력 artifacts를 모두 반환합니다. 여러 페이지 결과를 순회하거나 한 번에 리스트로 수집할 수 있습니다.



**인수:**
 
 - `per_page`:  API 요청당 가져올 artifact 개수



**반환값:**
 해당 run에서 output으로 기록된 모든 Artifact 오브젝트의 반복 가능한 컬렉션



**예시:**
 ```python
import wandb
import tempfile

with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
    tmp.write("This is a test artifact")
    tmp_path = tmp.name
run = wandb.init(project="artifact-example")
artifact = wandb.Artifact("test_artifact", type="dataset")
artifact.add_file(tmp_path)
run.log_artifact(artifact)
run.finish()

api = wandb.Api()

finished_run = api.run(f"{run.entity}/{run.project}/{run.id}")

for logged_artifact in finished_run.logged_artifacts():
    print(logged_artifact.name)
```

---

### <kbd>method</kbd> `Run.save`

```python
save()
```

run 오브젝트의 변경 사항을 W&B 백엔드에 저장합니다.

---

### <kbd>method</kbd> `Run.scan_history`

```python
scan_history(keys=None, page_size=1000, min_step=None, max_step=None)
```

run의 모든 히스토리 레코드를 반복 가능한 컬렉션으로 반환합니다.



**인수:**
 
 - `keys` ([str], optional):  해당 키만 가져오며, 모든 키가 정의된 행만 가져옵니다.
 - `page_size` (int, optional):  api에서 가져올 페이지 크기
 - `min_step` (int, optional):  한 번에 스캔할 최소 step 수
 - `max_step` (int, optional):  한 번에 스캔할 최대 step 수



**반환값:**
 히스토리 레코드(dict)를 순회할 수 있는 iterable 컬렉션



**예시:**
 예시 run의 loss 값 모두 추출

```python
run = api.run("entity/project-name/run-id")
history = run.scan_history(keys=["Loss"])
losses = [row["Loss"] for row in history]
```

---

### <kbd>method</kbd> `Run.to_html`

```python
to_html(height=420, hidden=False)
```

이 run을 표시하는 iframe이 포함된 HTML을 생성합니다.

---

### <kbd>method</kbd> `Run.update`

```python
update()
```

run 오브젝트의 변경 사항을 wandb 백엔드에 저장합니다.

---

### <kbd>method</kbd> `Run.upload_file`

```python
upload_file(path, root='.')
```

로컬 파일을 W&B로 업로드해 이 run에 연결합니다.



**인수:**
 
 - `path` (str):  업로드할 파일 경로 (절대/상대 경로 모두 가능)
 - `root` (str):  파일을 저장할 기준 경로. 예를 들어, run에서 파일을 "my_dir/file.txt" 형식으로 저장하고 현재 "my_dir" 디렉토리에 있다면 root를 "../"로 설정합니다. 기본값은 현재 디렉토리(".") 입니다.



**반환값:**
 업로드된 파일을 나타내는 `File` 오브젝트

---

### <kbd>method</kbd> `Run.use_artifact`

```python
use_artifact(artifact, use_as=None)
```

run의 입력으로 artifact를 선언합니다.



**인수:**
 
 - `artifact` (`Artifact`):  `wandb.Api().artifact(name)`에서 반환된 artifact
 - `use_as` (string, optional):  이 artifact가 스크립트에서 어떻게 사용되는지 식별하는 문자열. wandb launch 기능의 artifact 스와핑 기능 이용 시, run에 사용된 artifact를 쉽게 구분하는 데 사용됩니다.



**반환값:**
 `Artifact` 오브젝트 반환

---

### <kbd>method</kbd> `Run.used_artifacts`

```python
used_artifacts(per_page: int = 100) → RunArtifacts
```

이 run에서 명시적으로 사용된 artifact만을 가져옵니다.

주로 `run.use_artifact()`를 통해 명시적으로 선언된 입력 artifact만을 반환합니다. 여러 페이지 결과를 순회하거나 한 번에 리스트로 수집할 수 있습니다.



**인수:**
 
 - `per_page`:  API 요청당 가져올 artifact 개수



**반환값:**
 해당 run에서 입력으로 명시적으로 사용된 Artifact 오브젝트의 반복 가능한 컬렉션



**예시:**
 ```python
import wandb

run = wandb.init(project="artifact-example")
run.use_artifact("test_artifact:latest")
run.finish()

api = wandb.Api()
finished_run = api.run(f"{run.entity}/{run.project}/{run.id}")
for used_artifact in finished_run.used_artifacts():
    print(used_artifact.name)
test_artifact
```

---

### <kbd>method</kbd> `Run.wait_until_finished`

```python
wait_until_finished()
```

run이 완료될 때까지 상태를 확인합니다.
