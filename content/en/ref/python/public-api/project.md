---
title: Project
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/projects.py#L168-L278 >}}

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

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/projects.py#L247-L250)

```python
artifacts_types(
    per_page=50
)
```

Returns all artifact types associated with this project.

### `display`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/attrs.py#L18-L38)

```python
display(
    height=420, hidden=(False)
) -> bool
```

Display this object in jupyter.

### `snake_to_camel`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/attrs.py#L14-L16)

```python
snake_to_camel(
    string
)
```

### `sweeps`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/projects.py#L252-L262)

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

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/projects.py#L228-L239)

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
