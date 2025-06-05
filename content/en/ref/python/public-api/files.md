---
title: Files
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/files.py#L44-L107 >}}

An iterable collection of `File` objects.

```python
Files(
    client, run, names=None, per_page=50, upload=(False)
)
```

| Attributes |  |
| :--- | :--- |
|  `cursor` |  The start cursor to use for the next fetched page. |
|  `more` |  Whether there are more pages to be fetched. |

## Methods

### `convert_objects`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/files.py#L100-L104)

```python
convert_objects()
```

Convert the last fetched response data into the iterated objects.

### `next`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/paginator.py#L100-L107)

```python
next() -> T
```

Return the next item from the iterator. When exhausted, raise StopIteration

### `update_variables`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/files.py#L97-L98)

```python
update_variables()
```

Update the query variables for the next page fetch.

### `__getitem__`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/paginator.py#L93-L98)

```python
__getitem__(
    index: (int | slice)
) -> (T | list[T])
```

### `__iter__`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/paginator.py#L48-L50)

```python
__iter__() -> Iterator[T]
```

### `__len__`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/paginator.py#L115-L120)

```python
__len__() -> int
```

| Class Variables |  |
| :--- | :--- |
|  `QUERY`<a id="QUERY"></a> |   |
