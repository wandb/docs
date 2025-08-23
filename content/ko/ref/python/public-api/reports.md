---
title: 리포트
data_type_classification: module
menu:
  reference:
    identifier: ko-ref-python-public-api-reports
object_type: public_apis_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/reports.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API 는 Report 오브젝트를 위한 모듈입니다.

이 모듈은 W&B 리포트와 상호작용하고, 리포트 관련 데이터를 관리하기 위한 클래스를 제공합니다.



---

## <kbd>class</kbd> `Reports`
Reports 는 `BetaReport` 오브젝트의 iterable 컬렉션입니다.



**ARG:**

 - `client` (`wandb.apis.internal.Api`): 사용할 API client 인스턴스입니다.
 - `project` (`wandb.sdk.internal.Project`): 리포트를 가져올 프로젝트입니다.
 - `name` (str, optional): 필터링할 리포트 이름입니다. `None`이면 모든 리포트를 가져옵니다.
 - `entity` (str, optional): 프로젝트의 entity 이름입니다. 기본값은 프로젝트 entity 입니다.
 - `per_page` (int): 페이지당 불러올 리포트 수 (기본값은 50).

### <kbd>method</kbd> `Reports.__init__`

```python
__init__(client, project, name=None, entity=None, per_page=50)
```






---


### <kbd>property</kbd> Reports.length





---


### <kbd>method</kbd> `Reports.convert_objects`

```python
convert_objects()
```

GraphQL edge 를 File 오브젝트로 변환합니다.

---

### <kbd>method</kbd> `Reports.update_variables`

```python
update_variables()
```

페이지네이션을 위한 GraphQL 쿼리 변수들을 업데이트합니다.


---

## <kbd>class</kbd> `BetaReport`
BetaReport 는 W&B에서 생성된 리포트와 연결된 클래스입니다.

경고: 이 API는 향후 릴리즈에서 변경될 수 있습니다.



**속성:**

 - `id` (string): 리포트의 고유 식별자
 - `name` (string): 리포트 이름
 - `display_name` (string): 리포트의 표시 이름
 - `description` (string): 리포트 설명
 - `user` (User): 리포트를 생성한 사용자 (username과 email 포함)
 - `spec` (dict): 리포트의 spec
 - `url` (string): 리포트의 URL
 - `updated_at` (string): 마지막 업데이트 타임스탬프
 - `created_at` (string): 리포트가 생성된 시각의 타임스탬프

### <kbd>method</kbd> `BetaReport.__init__`

```python
__init__(client, attrs, entity=None, project=None)
```






---

### <kbd>property</kbd> BetaReport.created_at





---

### <kbd>property</kbd> BetaReport.description





---

### <kbd>property</kbd> BetaReport.display_name





---

### <kbd>property</kbd> BetaReport.id





---

### <kbd>property</kbd> BetaReport.name





---

### <kbd>property</kbd> BetaReport.sections

리포트에서 패널 섹션(그룹)들을 가져옵니다.

---

### <kbd>property</kbd> BetaReport.spec





---

### <kbd>property</kbd> BetaReport.updated_at





---

### <kbd>property</kbd> BetaReport.url





---

### <kbd>property</kbd> BetaReport.user







---

### <kbd>method</kbd> `BetaReport.runs`

```python
runs(section, per_page=50, only_selected=True)
```

리포트의 특정 섹션과 연결된 runs 를 가져옵니다.

---

### <kbd>method</kbd> `BetaReport.to_html`

```python
to_html(height=1024, hidden=False)
```

이 리포트를 표시하는 iframe이 포함된 HTML을 생성합니다.


---