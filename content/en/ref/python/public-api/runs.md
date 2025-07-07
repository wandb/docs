---
title: Runs
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/apis/public/runs.py#L88-L297 >}}

An iterable collection of runs associated with a project and optional filter.

This is generally used indirectly via the `Api`.runs method.

| Attributes |  |
| :--- | :--- |
|  `cursor` |  The start cursor to use for the next fetched page. |
|  `more` |  Whether there are more pages to be fetched. |

## Methods

### `convert_objects`

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/apis/public/runs.py#L165-L197)

```python
convert_objects()
```

Convert the last fetched response data into the iterated objects.

### `histories`

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/apis/public/runs.py#L199-L294)

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

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/apis/paginator.py#L102-L109)

```python
next() -> T
```

Return the next item from the iterator. When exhausted, raise StopIteration

### `update_variables`

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/apis/paginator.py#L71-L73)

```python
update_variables() -> None
```

Update the query variables for the next page fetch.

### `__getitem__`

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/apis/paginator.py#L95-L100)

```python
__getitem__(
    index: (int | slice)
) -> (T | list[T])
```

### `__iter__`

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/apis/paginator.py#L50-L52)

```python
__iter__() -> Iterator[T]
```

### `__len__`

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/apis/paginator.py#L128-L133)

```python
__len__() -> int
```

| Class Variables |  |
| :--- | :--- |
|  `QUERY`<a id="QUERY"></a> |  `None` |
