---
title: BetaReport
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.22.0/wandb/apis/public/reports.py#L141-L278 >}}

BetaReport is a class associated with reports created in W&B.

Provides access to report attributes (name, description, user, spec,
timestamps) and methods for retrieving associated runs,
sections, and for rendering the report as HTML.

| Attributes |  |
| :--- | :--- |
|  `sections` |  Get the panel sections (groups) from the report. |

## Methods

### `display`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.0/wandb/apis/attrs.py#L16-L36)

```python
display(
    height=420, hidden=(False)
) -> bool
```

Display this object in jupyter.

### `runs`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.0/wandb/apis/public/reports.py#L183-L205)

```python
runs(
    section, per_page=50, only_selected=(True)
)
```

Get runs associated with a section of the report.

### `snake_to_camel`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.0/wandb/apis/attrs.py#L12-L14)

```python
snake_to_camel(
    string
)
```

### `to_html`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.0/wandb/apis/public/reports.py#L264-L275)

```python
to_html(
    height=1024, hidden=(False)
)
```

Generate HTML containing an iframe displaying this report.
