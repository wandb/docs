# Runs

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/runs.py#L61-L269' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

프로젝트와 연관된 run의 이터러블 컬렉션이며, 선택적으로 필터를 추가할 수 있습니다.

```python
Runs(
    client: "RetryingClient",
    entity: str,
    project: str,
    filters: Optional[Dict[str, Any]] = None,
    order: Optional[str] = None,
    per_page: int = 50,
    include_sweeps: bool = (True)
)
```

이는 일반적으로 `Api`.runs 메소드를 통해 간접적으로 사용됩니다.

| 속성 |  |
| :--- | :--- |

## 메소드

### `convert_objects`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/runs.py#L136-L168)

```python
convert_objects()
```

### `histories`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/runs.py#L170-L266)

```python
histories(
    samples: int = 500,
    keys: Optional[List[str]] = None,
    x_axis: str = "_step",
    format: Literal['default', 'pandas', 'polars'] = "default",
    stream: Literal['default', 'system'] = "default"
)
```

필터 조건에 맞는 모든 run의 샘플링된 이력 메트릭을 반환합니다.

| 인수 |  |
| :--- | :--- |
|  `samples` |  (정수, 선택 항목) run당 반환할 샘플의 수 |
|  `keys` |  (리스트[str], 선택 항목) 특정 키에 대한 메트릭만 반환 |
|  `x_axis` |  (문자열, 선택 항목) x축으로 사용할 메트릭, 기본값은 _step |
|  `format` |  (Literal, 선택 항목) 데이터를 반환할 형식, "default", "pandas", "polars" 옵션 가능 |
|  `stream` |  (Literal, 선택 항목) 메트릭을 위한 "default", 시스템 메트릭을 위한 "system" |

| 반환 |  |
| :--- | :--- |
|  `pandas.DataFrame` |  format="pandas"일 경우, 이력 메트릭의 `pandas.DataFrame` 반환 |
|  `polars.DataFrame` |  format="polars"일 경우, 이력 메트릭의 `polars.DataFrame` 반환. list of dicts: format="default"일 경우, run_id 키를 포함하는 이력 메트릭의 사전 목록 반환. |

### `next`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/paginator.py#L72-L79)

```python
next()
```

### `update_variables`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/paginator.py#L52-L53)

```python
update_variables()
```

### `__getitem__`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/paginator.py#L65-L70)

```python
__getitem__(
    index
)
```

### `__iter__`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/paginator.py#L26-L28)

```python
__iter__()
```

### `__len__`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/paginator.py#L30-L35)

```python
__len__()
```

| 클래스 변수 |  |
| :--- | :--- |
|  `QUERY`<a id="QUERY"></a> |   |