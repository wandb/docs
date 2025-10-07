---
title: Projects
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.22.2/wandb/apis/public/projects.py#L56-L165 >}}

An lazy iterator of `Project` objects.

An iterable interface to access projects created and saved by the entity.

| Args |  |
| :--- | :--- |
|  client (`wandb.apis.internal.Api`): The API client instance to use. entity (str): The entity name (username or team) to fetch projects for. per_page (int): Number of projects to fetch per request (default is 50). |

#### Example:

```python
from wandb.apis.public.api import Api

# Find projects that belong to this entity
projects = Api().projects(entity="entity")

# Iterate over files
for project in projects:
    print(f"Project: {project.name}")
    print(f"- URL: {project.url}")
    print(f"- Created at: {project.created_at}")
    print(f"- Is benchmark: {project.is_benchmark}")
```

| Attributes |  |
| :--- | :--- |
|  `cursor` |  Returns the cursor position for pagination of project results. <!-- lazydoc-ignore: internal --> |
|  `length` |  Returns the total number of projects. Note: This property is not available for projects. <!-- lazydoc-ignore: internal --> |
|  `more` |  Returns `True` if there are more projects to fetch. Returns `False` if there are no more projects to fetch. <!-- lazydoc-ignore: internal --> |

## Methods

### `convert_objects`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.2/wandb/apis/public/projects.py#L154-L162)

```python
convert_objects()
```

Converts GraphQL edges to File objects.

<!-- lazydoc-ignore: internal -->


### `next`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.2/wandb/apis/paginator.py#L102-L109)

```python
next() -> T
```

Return the next item from the iterator. When exhausted, raise StopIteration

### `update_variables`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.2/wandb/apis/paginator.py#L71-L73)

```python
update_variables() -> None
```

Update the query variables for the next page fetch.

### `__getitem__`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.2/wandb/apis/paginator.py#L95-L100)

```python
__getitem__(
    index: (int | slice)
) -> (T | list[T])
```

### `__iter__`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.2/wandb/apis/paginator.py#L50-L52)

```python
__iter__() -> Iterator[T]
```

| Class Variables |  |
| :--- | :--- |
|  `QUERY`<a id="QUERY"></a> |   |
