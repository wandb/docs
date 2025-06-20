---
title: Project
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/projects.py#L83-L184 >}}

A project is a namespace for runs.

```python
Project(
    client, entity, project, attrs
)
```

| Attributes |  |
| :--- | :--- |

## Methods

### `artifacts_types`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/projects.py#L116-L118)

```python
artifacts_types(
    per_page=50
)
```

### `display`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/attrs.py#L16-L36)

```python
display(
    height=420, hidden=(False)
) -> bool
```

Display this object in jupyter.

### `snake_to_camel`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/attrs.py#L12-L14)

```python
snake_to_camel(
    string
)
```

### `sweeps`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/projects.py#L120-L158)

```python
sweeps()
```

### `to_html`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/projects.py#L100-L108)

```python
to_html(
    height=420, hidden=(False)
)
```

Generate HTML containing an iframe displaying this project.
