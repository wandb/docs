# Sweep

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/sweeps.py#L30-L240' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

스윕과 관련된 여러 run들의 집합입니다.

```python
Sweep(
    client, entity, project, sweep_id, attrs=None
)
```

#### 예시:

다음과 같이 인스턴스화:

```
api = wandb.Api()
sweep = api.sweep(path/to/sweep)
```

| 속성 |  |
| :--- | :--- |
|  `runs` |  (`Runs`) run들의 리스트 |
|  `id` |  (str) 스윕 id |
|  `project` |  (str) 프로젝트 이름 |
|  `config` |  (str) 스윕 구성 사전 |
|  `state` |  (str) 스윕의 상태 |
|  `expected_run_count` |  (int) 예상되는 run의 수 |

## 메소드

### `best_run`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/sweeps.py#L125-L148)

```python
best_run(
    order=None
)
```

구성에서 정의된 메트릭이나 전달된 순서에 따라 정렬된 최고의 run을 반환합니다.

### `display`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/attrs.py#L15-L26)

```python
display(
    height=420, hidden=(False)
) -> bool
```

이 오브젝트를 주피터에서 표시합니다.

### `get`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/sweeps.py#L173-L222)

```python
@classmethod
get(
    client, entity=None, project=None, sid=None, order=None, query=None, **kwargs
)
```

클라우드 백엔드에 대해 쿼리를 실행합니다.

### `load`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/sweeps.py#L106-L114)

```python
load(
    force: bool = (False)
)
```

### `snake_to_camel`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/attrs.py#L11-L13)

```python
snake_to_camel(
    string
)
```

### `to_html`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/sweeps.py#L224-L232)

```python
to_html(
    height=420, hidden=(False)
)
```

이 스윕을 표시하는 iframe을 포함한 HTML을 생성합니다.

| 클래스 변수 |  |
| :--- | :--- |
|  `LEGACY_QUERY`<a id="LEGACY_QUERY"></a> |   |
|  `QUERY`<a id="QUERY"></a> |   |