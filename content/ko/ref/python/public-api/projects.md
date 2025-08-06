---
title: 프로젝트
data_type_classification: module
menu:
  reference:
    identifier: ko-ref-python-public-api-projects
object_type: public_apis_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/projects.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API에서 Project 오브젝트를 다루는 모듈입니다.

이 모듈은 W&B Projects와 그에 연관된 데이터와 상호작용할 수 있는 클래스를 제공합니다.



**예시:**
 ```python
from wandb.apis.public import Api

# 엔티티의 모든 프로젝트 가져오기
projects = Api().projects("entity")

# 프로젝트 데이터 엑세스
for project in projects:
     print(f"Project: {project.name}")
     print(f"URL: {project.url}")

     # 아티팩트 타입 가져오기
     for artifact_type in project.artifacts_types():
         print(f"Artifact Type: {artifact_type.name}")

     # 스윕 가져오기
     for sweep in project.sweeps():
         print(f"Sweep ID: {sweep.id}")
         print(f"State: {sweep.state}")
```



**노트:**

> 이 모듈은 W&B Public API의 일부로, Projects에 엑세스하고 관리하는 메소드를 제공합니다. 새로운 프로젝트를 만들려면 wandb.init()에서 새로운 프로젝트 이름을 사용하세요.

## <kbd>class</kbd> `Projects`
`Project` 오브젝트의 반복 가능한 컬렉션입니다.

엔티티가 만든 Projects에 반복적으로 접근할 수 있는 인터페이스입니다.



**인자:**
 
 - `client` (`wandb.apis.internal.Api`): 사용할 API 클라이언스 인스턴스
 - `entity` (str): 프로젝트를 조회할 엔티티 이름 (사용자명 또는 팀명)
 - `per_page` (int): 한 번에 요청해서 가져올 프로젝트 수 (기본값: 50)



**예시:**
 ```python
from wandb.apis.public.api import Api

# 해당 엔티티에 속한 프로젝트 찾기
projects = Api().projects(entity="entity")

# 파일 반복
for project in projects:
    print(f"Project: {project.name}")
    print(f"- URL: {project.url}")
    print(f"- Created at: {project.created_at}")
    print(f"- Is benchmark: {project.is_benchmark}")
```

### <kbd>method</kbd> `Projects.__init__`

```python
__init__(
    client: wandb.apis.public.api.RetryingClient,
    entity: str,
    per_page: int = 50
) → Projects
```

`Project` 오브젝트의 반복 가능한 컬렉션입니다.



**인자:**
 
 - `client`:  W&B를 쿼리하는 데 사용되는 API 클라이언트
 - `entity`:  Projects를 소유한 엔티티
 - `per_page`:  API에 한 번에 요청-조회할 Projects의 개수


---





## <kbd>class</kbd> `Project`
프로젝트는 run들을 위한 네임스페이스입니다.



**인자:**
 
 - `client`:  W&B API 클라이언트 인스턴스
 - `name` (str):  프로젝트 이름
 - `entity` (str):  해당 프로젝트의 소유 엔티티 이름

### <kbd>method</kbd> `Project.__init__`

```python
__init__(
    client: wandb.apis.public.api.RetryingClient,
    entity: str,
    project: str,
    attrs: dict
) → Project
```

엔티티와 연결된 하나의 프로젝트입니다.



**인자:**
 
 - `client`:  W&B를 쿼리하는 데 사용되는 API 클라이언트
 - `entity`:  프로젝트를 소유한 엔티티
 - `project`:  조회할 프로젝트 이름
 - `attrs`:  프로젝트의 속성들


---

### <kbd>property</kbd> Project.id





---

### <kbd>property</kbd> Project.path

프로젝트의 경로를 반환합니다. 경로는 엔티티와 프로젝트 이름이 담긴 리스트입니다.

---

### <kbd>property</kbd> Project.url

프로젝트의 URL을 반환합니다.



---

### <kbd>method</kbd> `Project.artifacts_types`

```python
artifacts_types(per_page=50)
```

이 프로젝트와 연관된 모든 artifact 타입을 반환합니다.

---

### <kbd>method</kbd> `Project.sweeps`

```python
sweeps()
```

해당 프로젝트와 연관된 모든 sweeps를 조회합니다.

---