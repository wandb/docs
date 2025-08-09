---
title: BetaReport
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.21.1/wandb/apis/public/reports.py#L141-L275 >}}

BetaReport is a class associated with reports created in W&B.

WARNING: this API will likely change in a future release

| Attributes |  |
| :--- | :--- |
|  `sections` |  Get the panel sections (groups) from the report. |

## Methods

### `display`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.1/wandb/apis/attrs.py#L16-L36)

```python
display(
    height=420, hidden=(False)
) -> bool
```

Display this object in jupyter.

### `runs`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.1/wandb/apis/public/reports.py#L180-L202)

```python
runs(
    section, per_page=50, only_selected=(True)
)
```

Get runs associated with a section of the report.

### `snake_to_camel`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.1/wandb/apis/attrs.py#L12-L14)

```python
snake_to_camel(
    string
)
```

### `to_html`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.1/wandb/apis/public/reports.py#L261-L272)

```python
to_html(
    height=1024, hidden=(False)
)
```

Generate HTML containing an iframe displaying this report.
