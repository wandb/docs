---
title: BetaReport
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.22.2/wandb/apis/public/reports.py#L143-L280 >}}

BetaReport is a class associated with reports created in W&B.

Provides access to report attributes (name, description, user, spec,
timestamps) and methods for retrieving associated runs,
sections, and for rendering the report as HTML.

| Attributes |  |
| :--- | :--- |
|  `sections` |  Get the panel sections (groups) from the report. |

## Methods

### `display`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.2/wandb/apis/attrs.py#L18-L38)

```python
display(
    height=420, hidden=(False)
) -> bool
```

Display this object in jupyter.

### `runs`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.2/wandb/apis/public/reports.py#L185-L207)

```python
runs(
    section, per_page=50, only_selected=(True)
)
```

Get runs associated with a section of the report.

### `snake_to_camel`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.2/wandb/apis/attrs.py#L14-L16)

```python
snake_to_camel(
    string
)
```

### `to_html`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.2/wandb/apis/public/reports.py#L266-L277)

```python
to_html(
    height=1024, hidden=(False)
)
```

Generate HTML containing an iframe displaying this report.
