---
title: Project
---

{{< cta-button githubLink="https://www.github.com/wandb/wandb/tree/v0.18.7/wandb/apis/public/projects.py#L79-L154" >}}

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

[View source](https://www.github.com/wandb/wandb/tree/v0.19.1/wandb/apis/public/projects.py#L112-L114)

```python
artifacts_types(
    per_page=50
)
```

### `display`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.1/wandb/apis/attrs.py#L16-L37)

```python
display(
    height=420, hidden=(False)
) -> bool
```

Display this object in jupyter.

### `snake_to_camel`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.1/wandb/apis/attrs.py#L12-L14)

```python
snake_to_camel(
    string
)
```

### `sweeps`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.1/wandb/apis/public/projects.py#L116-L154)

```python
sweeps()
```

### `to_html`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.1/wandb/apis/public/projects.py#L96-L104)

```python
to_html(
    height=420, hidden=(False)
)
```

Generate HTML containing an iframe displaying this project.
