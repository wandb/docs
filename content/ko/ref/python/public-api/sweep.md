---
title: Sweep
menu:
  reference:
    identifier: ko-ref-python-public-api-sweep
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/sweeps.py#L30-L240 >}}

스윕과 관련된 run들의 집합입니다.

```python
Sweep(
    client, entity, project, sweep_id, attrs=None
)
```

#### 예시:

다음과 같이 인스턴스화합니다:

```
api = wandb.Api()
sweep = api.sweep(path / to / sweep)
```

| 속성 |  |
| :--- | :--- |
|  `runs` |  (`Runs`) run들의 목록 |
|  `id` |  (str) 스윕 ID |
|  `project` |  (str) 프로젝트 이름 |
|  `config` |  (str) 스윕 구성의 사전 |
|  `state` |  (str) 스윕의 상태 |
|  `expected_run_count` |  (int) 스윕에 예상되는 run의 수 |

## 메소드

### `best_run`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/sweeps.py#L125-L148)

```python
best_run(
    order=None
)
```

구성에서 정의된 메트릭 또는 전달된 순서에 따라 정렬된 최상의 run을 반환합니다.

### `display`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/attrs.py#L16-L37)

```python
display(
    height=420, hidden=(False)
) -> bool
```

이 오브젝트를 jupyter에 표시합니다.

### `get`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/sweeps.py#L173-L222)

```python
@classmethod
get(
    client, entity=None, project=None, sid=None, order=None, query=None, **kwargs
)
```

클라우드 백엔드에 대해 쿼리를 실행합니다.

### `load`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/sweeps.py#L106-L114)

```python
load(
    force: bool = (False)
)
```

### `snake_to_camel`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/attrs.py#L12-L14)

```python
snake_to_camel(
    string
)
```

### `to_html`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/sweeps.py#L224-L232)

```python
to_html(
    height=420, hidden=(False)
)
```

이 스윕을 표시하는 iframe을 포함하는 HTML을 생성합니다.

| 클래스 변수 |  |
| :--- | :--- |
|  `LEGACY_QUERY`<a id="LEGACY_QUERY"></a> |   |
|  `QUERY`<a id="QUERY"></a> |   |
