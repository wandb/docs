---
title: Files
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/apis/public/files.py#L44-L107 >}}

An iterable collection of `File` objects.

| Attributes |  |
| :--- | :--- |
|  `cursor` |  The start cursor to use for the next fetched page. |
|  `more` |  Whether there are more pages to be fetched. |

## Methods

### `convert_objects`

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/apis/public/files.py#L100-L104)

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

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/apis/public/files.py#L97-L98)

```python
update_variables()
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
|  `QUERY`<a id="QUERY"></a> |   |
