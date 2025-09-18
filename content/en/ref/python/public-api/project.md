---
title: Project
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.22.0/wandb/apis/public/projects.py#L166-L276 >}}

A project is a namespace for runs.

| Args |  |
| :--- | :--- |
|  `client` |  W&B API client instance. name (str): The name of the project. entity (str): The entity name that owns the project. |

| Attributes |  |
| :--- | :--- |
|  `path` |  Returns the path of the project. The path is a list containing the entity and project name. |
|  `url` |  Returns the URL of the project. |

## Methods

### `artifacts_types`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.0/wandb/apis/public/projects.py#L245-L248)

```python
artifacts_types(
    per_page=50
)
```

Returns all artifact types associated with this project.

### `display`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.0/wandb/apis/attrs.py#L16-L36)

```python
display(
    height=420, hidden=(False)
) -> bool
```

Display this object in jupyter.

### `snake_to_camel`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.0/wandb/apis/attrs.py#L12-L14)

```python
snake_to_camel(
    string
)
```

### `sweeps`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.0/wandb/apis/public/projects.py#L250-L260)

```python
sweeps(
    per_page=50
)
```

Return a paginated collection of sweeps in this project.

| Args |  |
| :--- | :--- |
|  `per_page` |  The number of sweeps to fetch per request to the API. |

| Returns |  |
| :--- | :--- |
|  A `Sweeps` object, which is an iterable collection of `Sweep` objects. |

### `to_html`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.0/wandb/apis/public/projects.py#L226-L237)

```python
to_html(
    height=420, hidden=(False)
)
```

Generate HTML containing an iframe displaying this project.

<!-- lazydoc-ignore: internal -->


| Class Variables |  |
| :--- | :--- |
|  `QUERY`<a id="QUERY"></a> |   |
