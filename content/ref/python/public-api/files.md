---
title: Files
---

{{< cta-button githubLink="https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/apis/public/files.py#L44-L106" >}}

An iterable collection of `File` objects.

```python
Files(
    client, run, names=None, per_page=50, upload=(False)
)
```

| Attributes |  |
| :--- | :--- |

## Methods

### `convert_objects`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/apis/public/files.py#L100-L104)

```python
convert_objects()
```

### `next`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/apis/paginator.py#L72-L79)

```python
next()
```

### `update_variables`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/apis/public/files.py#L97-L98)

```python
update_variables()
```

### `__getitem__`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/apis/paginator.py#L65-L70)

```python
__getitem__(
    index
)
```

### `__iter__`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/apis/paginator.py#L26-L28)

```python
__iter__()
```

### `__len__`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/apis/paginator.py#L30-L35)

```python
__len__()
```

| Class Variables |  |
| :--- | :--- |
|  `QUERY`<a id="QUERY"></a> |   |
