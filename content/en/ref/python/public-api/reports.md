---
title: Reports
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.21.1/wandb/apis/public/reports.py#L21-L138 >}}

Reports is a lazy iterator of `BetaReport` objects.

| Args |  |
| :--- | :--- |
|  client (`wandb.apis.internal.Api`): The API client instance to use. project (`wandb.sdk.internal.Project`): The project to fetch reports from. name (str, optional): The name of the report to filter by. If `None`, fetches all reports. entity (str, optional): The entity name for the project. Defaults to the project entity. per_page (int): Number of reports to fetch per page (default is 50). |

| Attributes |  |
| :--- | :--- |
|  `cursor` |  Returns the cursor position for pagination of file results. <!-- lazydoc-ignore: internal --> |
|  `more` |  Returns whether there are more files to fetch. <!-- lazydoc-ignore: internal --> |

## Methods

### `convert_objects`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.1/wandb/apis/public/reports.py#L121-L135)

```python
convert_objects()
```

Converts GraphQL edges to File objects.

### `next`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.1/wandb/apis/paginator.py#L102-L109)

```python
next() -> T
```

Return the next item from the iterator. When exhausted, raise StopIteration

### `update_variables`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.1/wandb/apis/public/reports.py#L115-L119)

```python
update_variables()
```

Updates the GraphQL query variables for pagination.

### `__getitem__`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.1/wandb/apis/paginator.py#L95-L100)

```python
__getitem__(
    index: (int | slice)
) -> (T | list[T])
```

### `__iter__`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.1/wandb/apis/paginator.py#L50-L52)

```python
__iter__() -> Iterator[T]
```

### `__len__`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.1/wandb/apis/paginator.py#L128-L133)

```python
__len__() -> int
```

| Class Variables |  |
| :--- | :--- |
|  `QUERY`<a id="QUERY"></a> |   |
