---
title: 스윕
data_type_classification: module
menu:
  reference:
    identifier: ko-ref-python-public-api-sweeps
object_type: public_apis_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/sweeps.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API for Sweeps.

이 모듈은 W&B 하이퍼파라미터 최적화 Sweeps 와 상호작용할 수 있는 클래스를 제공합니다.


**예시:**
 ```python
from wandb.apis.public import Api

# 특정 sweep 가져오기
sweep = Api().sweep("entity/project/sweep_id")

# sweep 속성 엑세스
print(f"Sweep: {sweep.name}")
print(f"State: {sweep.state}")
print(f"Best Loss: {sweep.best_loss}")

# 성능이 가장 좋은 run 가져오기
best_run = sweep.best_run()
print(f"Best Run: {best_run.name}")
print(f"Metrics: {best_run.summary}")
```

**참고:**

> 이 모듈은 W&B Public API 의 일부로, sweep 데이터에 대한 읽기 전용 엑세스를 제공합니다. Sweeps 를 생성하거나 제어하고 싶다면, 메인 wandb 패키지의 wandb.sweep() 및 wandb.agent() 함수를 사용하세요.

## <kbd>class</kbd> `Sweep`
해당 sweep 와 연관된 run 들의 집합입니다.


**속성:**

 - `runs` (Runs):  run 목록
 - `id` (str):  Sweep ID
 - `project` (str):  이 sweep 가 속한 프로젝트 이름
 - `config` (dict): 스윕 구성이 담긴 사전
 - `state` (str):  Sweep 의 상태. "Finished", "Failed", "Crashed", "Running" 중 하나일 수 있습니다.
 - `expected_run_count` (int):  이 sweep 에서 실행될 것으로 예상되는 run 개수

### <kbd>method</kbd> `Sweep.__init__`

```python
__init__(client, entity, project, sweep_id, attrs=None)
```



---

### <kbd>property</kbd> Sweep.config

스윕에 사용된 스윕 구성입니다.

---

### <kbd>property</kbd> Sweep.entity

이 sweep 에 연관된 entity 를 반환합니다.

---

### <kbd>property</kbd> Sweep.expected_run_count

스윕에서 예상되는 run 개수를 반환합니다. 무한 run 인 경우 None 을 반환합니다.

---

### <kbd>property</kbd> Sweep.name

스윕의 이름입니다.

스윕에 이름이 지정되어 있으면 해당 이름이 반환되고, 그렇지 않으면 Sweep ID 가 반환됩니다.

---

### <kbd>property</kbd> Sweep.order

스윕의 order 키를 반환합니다.

---

### <kbd>property</kbd> Sweep.path

프로젝트의 경로를 반환합니다.

경로는 entity, 프로젝트 이름, sweep ID 로 구성된 리스트입니다.

---

### <kbd>property</kbd> Sweep.url

스윕의 URL입니다.

스윕 URL은 entity, 프로젝트, "sweeps"라는 용어, 그리고 sweep ID.run_id를 이용해 생성됩니다. SaaS 사용자라면 `https://wandb.ai/entity/project/sweeps/sweeps_ID` 형태로 표시됩니다.

---

### <kbd>property</kbd> Sweep.username

더 이상 사용되지 않습니다. 대신 `Sweep.entity` 를 사용하세요.


---

### <kbd>method</kbd> `Sweep.best_run`

```python
best_run(order=None)
```

설정에서 정의된 메트릭 또는 입력한 order 기준으로 정렬해 가장 성능이 좋은 run 을 반환합니다.

---

### <kbd>classmethod</kbd> `Sweep.get`

```python
get(
    client,
    entity=None,
    project=None,
    sid=None,
    order=None,
    query=None,
    **kwargs
)
```

클라우드 백엔드에 쿼리를 실행합니다.

---


### <kbd>method</kbd> `Sweep.to_html`

```python
to_html(height=420, hidden=False)
```

이 sweep 을 표시하는 iframe 이 포함된 HTML 을 생성합니다.