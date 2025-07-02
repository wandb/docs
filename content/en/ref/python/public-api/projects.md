---
title: Projects
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/apis/public/projects.py#L24-L93 >}}

An iterable collection of `Project` objects.

| Attributes |  |
| :--- | :--- |
|  `cursor` |  The start cursor to use for the next fetched page. |
|  `more` |  Whether there are more pages to be fetched. |

## Methods

### `convert_objects`

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/apis/public/projects.py#L86-L90)

```python
convert_objects()
```

Convert the last fetched response data into the iterated objects.

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

| Class Variables |  |
| :--- | :--- |
|  `QUERY`<a id="QUERY"></a> |   |
