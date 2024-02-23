
# 스윕

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/sweeps.py#L29-L239' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>

스윕과 관련된 실행들의 집합입니다.

```python
Sweep(
    client, entity, project, sweep_id, attrs=None
)
```

#### 예시:

다음과 같이 인스턴스를 생성합니다.

```
api = wandb.Api()
sweep = api.sweep(path/to/sweep)
```

| 속성 |  |
| :--- | :--- |
|  `runs` |  (`Runs`) 실행 목록 |
|  `id` |  (str) 스윕 ID |
|  `project` |  (str) 프로젝트 이름 |
|  `config` |  (str) 스윕 구성의 사전 |
|  `state` |  (str) 스윕의 상태 |
|  `expected_run_count` |  (int) 스윕을 위한 예상 실행 횟수 |

## 메서드

### `best_run`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/sweeps.py#L124-L147)

```python
best_run(
    order=None
)
```

구성에서 정의된 메트릭 또는 전달된 순서에 따라 정렬된 최상의 실행을 반환합니다.

### `display`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/attrs.py#L15-L26)

```python
display(
    height=420, hidden=(False)
) -> bool
```

이 개체를 jupyter에 표시합니다.

### `get`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/sweeps.py#L172-L221)

```python
@classmethod
get(
    client, entity=None, project=None, sid=None, order=None, query=None, **kwargs
)
```

클라우드 백엔드에 대한 쿼리를 실행합니다.

### `load`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/sweeps.py#L105-L113)

```python
load(
    force: bool = (False)
)
```

### `snake_to_camel`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/attrs.py#L11-L13)

```python
snake_to_camel(
    string
)
```

### `to_html`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/sweeps.py#L223-L231)

```python
to_html(
    height=420, hidden=(False)
)
```

이 스윕을 표시하는 iframe이 포함된 HTML을 생성합니다.

| 클래스 변수 |  |
| :--- | :--- |
|  `LEGACY_QUERY`<a id="LEGACY_QUERY"></a> |   |
|  `QUERY`<a id="QUERY"></a> |   |