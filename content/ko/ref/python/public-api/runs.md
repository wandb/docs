---
title: Runs
menu:
  reference:
    identifier: ko-ref-python-public-api-runs
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/runs.py#L64-L273 >}}

프로젝트 및 선택적 필터와 연결된 runs의 반복 가능한 컬렉션입니다.

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

일반적으로 `Api`.runs 메소드를 통해 간접적으로 사용됩니다.

| 속성 |  |
| :--- | :--- |

## 메소드

### `convert_objects`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/runs.py#L141-L173)

```python
convert_objects()
```

### `histories`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/runs.py#L175-L270)

```python
histories(
    samples: int = 500,
    keys: Optional[List[str]] = None,
    x_axis: str = "_step",
    format: Literal['default', 'pandas', 'polars'] = "default",
    stream: Literal['default', 'system'] = "default"
)
```

필터 조건에 맞는 모든 runs에 대해 샘플링된 히스토리 메트릭을 반환합니다.

| Args |  |
| :--- | :--- |
|  `samples` |  (int, optional) run 당 반환할 샘플 수 |
|  `keys` |  (list[str], optional) 특정 키에 대한 메트릭만 반환 |
|  `x_axis` |  (str, optional) 이 메트릭을 xAxis로 사용, 기본값은 _step |
|  `format` |  (Literal, optional) 데이터를 반환할 형식, 옵션은 "default", "pandas", "polars" |
|  `stream` |  (Literal, optional) 메트릭의 경우 "default", 머신 메트릭의 경우 "system" |

| 반환 값 |  |
| :--- | :--- |
|  `pandas.DataFrame` |  format="pandas"인 경우, 히스토리 메트릭의 `pandas.DataFrame`을 반환합니다. |
|  `polars.DataFrame` |  format="polars"인 경우, 히스토리 메트릭의 `polars.DataFrame`을 반환합니다. list of dicts: format="default"인 경우, run_id 키가 있는 히스토리 메트릭을 포함하는 dicts 목록을 반환합니다. |

### `next`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/paginator.py#L72-L79)

```python
next()
```

### `update_variables`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/paginator.py#L52-L53)

```python
update_variables()
```

### `__getitem__`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/paginator.py#L65-L70)

```python
__getitem__(
    index
)
```

### `__iter__`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/paginator.py#L26-L28)

```python
__iter__()
```

### `__len__`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/paginator.py#L30-L35)

```python
__len__()
```

| 클래스 변수 |  |
| :--- | :--- |
|  `QUERY`<a id="QUERY"></a> |   |
