---
title: 아티팩트
data_type_classification: module
menu:
  reference:
    identifier: ko-ref-python-public-api-artifacts
object_type: public_apis_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/artifacts.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API에서 Artifact 오브젝트를 다루는 모듈입니다.

이 모듈은 W&B Artifacts 및 그 컬렉션과 상호작용할 수 있는 클래스들을 제공합니다.


## <kbd>class</kbd> `ArtifactTypes`
특정 프로젝트에 대한 artifact 타입들의 순회 가능한 컬렉션입니다.


## <kbd>class</kbd> `ArtifactType`
지정된 타입에 기반하여 쿼리 조건을 만족하는 artifact 오브젝트입니다.


**인자:**
 
 - `client`:  W&B 쿼리에 사용할 클라이언트 인스턴스입니다.
 - `entity`:  프로젝트를 소유한 Entity (user 또는 team)입니다.
 - `project`:  artifact 타입을 쿼리할 프로젝트명입니다.
 - `type_name`:  artifact 타입명입니다.
 - `attrs`:  artifact 타입을 초기화할 속성의 선택적 매핑입니다. 제공되지 않으면 이 오브젝트는 초기화 시 W&B에서 속성값을 불러옵니다.


### <kbd>property</kbd> ArtifactType.id

아티팩트 타입의 고유 식별자입니다.

---

### <kbd>property</kbd> ArtifactType.name

아티팩트 타입의 이름입니다.



---

### <kbd>method</kbd> `ArtifactType.collection`

```python
collection(name: 'str') → ArtifactCollection
```

이름으로 특정 artifact 컬렉션을 가져옵니다.



**인자:**
 
 - `name` (str):  가져올 artifact 컬렉션의 이름입니다.

---

### <kbd>method</kbd> `ArtifactType.collections`

```python
collections(per_page: 'int' = 50) → ArtifactCollections
```

이 artifact 타입과 연관된 모든 artifact 컬렉션을 가져옵니다.



**인자:**
 
 - `per_page` (int):  한 페이지에 불러올 artifact 컬렉션 개수입니다. 기본값은 50입니다.

---


## <kbd>class</kbd> `ArtifactCollections`
프로젝트 내 특정 타입에 해당하는 artifact 컬렉션 객체입니다.


**인자:**
 
 - `client`:  W&B 쿼리에 사용할 클라이언트 인스턴스입니다.
 - `entity`:  프로젝트를 소유한 Entity (user 또는 team)입니다.
 - `project`:  artifact 컬렉션을 쿼리할 프로젝트명입니다.
 - `type_name`:  컬렉션을 가져올 artifact 타입명입니다.
 - `per_page`:  한 페이지에 불러올 artifact 컬렉션 개수입니다. 기본값은 50입니다.


### <kbd>property</kbd> ArtifactCollections.length





---




## <kbd>class</kbd> `ArtifactCollection`
관련된 artifacts 를 그룹화하는 artifact 컬렉션입니다.


**인자:**
 
 - `client`:  W&B 쿼리에 사용할 클라이언트 인스턴스입니다.
 - `entity`:  프로젝트를 소유한 Entity (user 또는 team)입니다.
 - `project`:  artifact 컬렉션을 쿼리할 프로젝트명입니다.
 - `name`:  artifact 컬렉션의 이름입니다.
 - `type`:  artifact 컬렉션의 타입 (예: "dataset", "model").
 - `organization`:  필요하다면 적용할 조직명 (선택 사항).
 - `attrs`:  artifact 컬렉션을 초기화할 속성의 선택적 매핑입니다. 제공되지 않으면 이 오브젝트는 초기화 시 W&B에서 속성값을 불러옵니다.


### <kbd>property</kbd> ArtifactCollection.aliases

Artifact Collection의 에일리어스입니다.

---

### <kbd>property</kbd> ArtifactCollection.created_at

아티팩트 컬렉션의 생성 날짜입니다.

---

### <kbd>property</kbd> ArtifactCollection.description

아티팩트 컬렉션에 대한 설명입니다.

---

### <kbd>property</kbd> ArtifactCollection.id

아티팩트 컬렉션의 고유 식별자입니다.

---

### <kbd>property</kbd> ArtifactCollection.name

아티팩트 컬렉션의 이름입니다.

---

### <kbd>property</kbd> ArtifactCollection.tags

아티팩트 컬렉션에 연결된 태그들입니다.

---

### <kbd>property</kbd> ArtifactCollection.type

아티팩트 컬렉션의 타입을 반환합니다.



---

### <kbd>method</kbd> `ArtifactCollection.artifacts`

```python
artifacts(per_page: 'int' = 50) → Artifacts
```

이 컬렉션에 포함된 모든 artifacts 를 가져옵니다.

---

### <kbd>method</kbd> `ArtifactCollection.change_type`

```python
change_type(new_type: 'str') → None
```

더 이상 사용되지 않으며, 타입 변경은 `save`를 통해 직접 처리하세요.

---

### <kbd>method</kbd> `ArtifactCollection.delete`

```python
delete() → None
```

전체 artifact 컬렉션을 삭제합니다.

---

### <kbd>method</kbd> `ArtifactCollection.is_sequence`

```python
is_sequence() → bool
```

해당 artifact 컬렉션이 시퀀스(연속적인 값)인지 여부를 반환합니다.

---


### <kbd>method</kbd> `ArtifactCollection.save`

```python
save() → None
```

artifact 컬렉션에 대한 변경사항을 영구적으로 저장합니다.


---

## <kbd>class</kbd> `Artifacts`
프로젝트에 연결된 artifact 버전의 순회 가능한 컬렉션입니다.

특정 기준을 기반으로 결과를 좁힐 수 있도록 필터를 전달할 수 있습니다.


**인자:**
 
 - `client`:  W&B 쿼리에 사용할 클라이언트 인스턴스입니다.
 - `entity`:  프로젝트를 소유한 Entity (user 또는 team)입니다.
 - `project`:  artifacts 를 쿼리할 프로젝트 이름입니다.
 - `collection_name`:  쿼리할 artifact 컬렉션명입니다.
 - `type`:  쿼리할 artifacts 타입입니다. 예시로 "dataset", "model" 등.
 - `filters`:  쿼리에 적용할 선택적 필터 매핑입니다.
 - `order`:  결과의 정렬 순서를 지정할 선택적 문자열입니다.
 - `per_page`:  한 페이지에 불러올 artifact 버전 개수입니다. 기본값은 50입니다.
 - `tags`:  태그별로 artifacts 를 필터링할 때 사용할 선택적 문자열 또는 문자열 리스트입니다.


### <kbd>property</kbd> Artifacts.length





---



## <kbd>class</kbd> `RunArtifacts`
특정 run 과 연결된 artifacts 의 순회 가능한 컬렉션입니다.


### <kbd>property</kbd> RunArtifacts.length





---



## <kbd>class</kbd> `ArtifactFiles`
artifact 내 파일들의 페이지네이터입니다.


### <kbd>property</kbd> ArtifactFiles.length





---


### <kbd>property</kbd> ArtifactFiles.path

아티팩트의 경로를 반환합니다.



---