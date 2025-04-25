---
title: Runs
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.19.10/wandb/apis/public/runs.py#L64-L273 >}}

An iterable collection of runs associated with a project and optional filter.

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

This is generally used indirectly via the `Api`.runs method.

| Attributes |  |
| :--- | :--- |
|  `cursor` |  The start cursor to use for the next fetched page. |
|  `more` |  Whether there are more pages to be fetched. |

## Methods

### `convert_objects`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.10/wandb/apis/public/runs.py#L141-L173)

```python
convert_objects()
```

Convert the last fetched response data into the iterated objects.

### `histories`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.10/wandb/apis/public/runs.py#L175-L270)

```python
histories(
    samples: int = 500,
    keys: Optional[List[str]] = None,
    x_axis: str = "_step",
    format: Literal['default', 'pandas', 'polars'] = "default",
    stream: Literal['default', 'system'] = "default"
)
```

Return sampled history metrics for all runs that fit the filters conditions.

| Args |  |
| :--- | :--- |
|  `samples` |  (int, optional) The number of samples to return per run |
|  `keys` |  (list[str], optional) Only return metrics for specific keys |
|  `x_axis` |  (str, optional) Use this metric as the xAxis defaults to _step |
|  `format` |  (Literal, optional) Format to return data in, options are "default", "pandas", "polars" |
|  `stream` |  (Literal, optional) "default" for metrics, "system" for machine metrics |

| Returns |  |
| :--- | :--- |
|  `pandas.DataFrame` |  If format="pandas", returns a `pandas.DataFrame` of history metrics. |
|  `polars.DataFrame` |  If format="polars", returns a `polars.DataFrame` of history metrics. list of dicts: If format="default", returns a list of dicts containing history metrics with a run_id key. |

### `next`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.10/wandb/apis/paginator.py#L100-L107)

```python
next() -> T
```

Return the next item from the iterator. When exhausted, raise StopIteration

### `update_variables`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.10/wandb/apis/paginator.py#L69-L71)

```python
update_variables() -> None
```

Update the query variables for the next page fetch.

### `__getitem__`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.10/wandb/apis/paginator.py#L93-L98)

```python
__getitem__(
    index: (int | slice)
) -> (T | list[T])
```

### `__iter__`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.10/wandb/apis/paginator.py#L48-L50)

```python
__iter__() -> Iterator[T]
```

### `__len__`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.10/wandb/apis/paginator.py#L115-L120)

```python
__len__() -> int
```

| Class Variables |  |
| :--- | :--- |
|  `QUERY`<a id="QUERY"></a> |   |
