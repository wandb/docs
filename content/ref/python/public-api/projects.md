---
title: Projects
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.19.9/wandb/apis/public/projects.py#L20-L77 >}}

An iterable collection of `Project` objects.

```python
Projects(
    client, entity, per_page=50
)
```

| Attributes |  |
| :--- | :--- |
|  `cursor` |  The start cursor to use for the next fetched page. |
|  `more` |  Whether there are more pages to be fetched. |

## Methods

### `convert_objects`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.9/wandb/apis/public/projects.py#L70-L74)

```python
convert_objects()
```

Convert the last fetched response data into the iterated objects.

### `next`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.9/wandb/apis/paginator.py#L100-L107)

```python
next() -> T
```

Return the next item from the iterator. When exhausted, raise StopIteration

### `update_variables`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.9/wandb/apis/paginator.py#L69-L71)

```python
update_variables() -> None
```

Update the query variables for the next page fetch.

### `__getitem__`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.9/wandb/apis/paginator.py#L93-L98)

```python
__getitem__(
    index: (int | slice)
) -> (T | list[T])
```

### `__iter__`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.9/wandb/apis/paginator.py#L48-L50)

```python
__iter__() -> Iterator[T]
```

| Class Variables |  |
| :--- | :--- |
|  `QUERY`<a id="QUERY"></a> |   |
