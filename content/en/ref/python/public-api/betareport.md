---
title: BetaReport
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/apis/public/reports.py#L111-L243 >}}

BetaReport is a class associated with reports created in wandb.

WARNING: this API will likely change in a future release

| Attributes |  |
| :--- | :--- |

## Methods

### `display`

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/apis/attrs.py#L16-L36)

```python
display(
    height=420, hidden=(False)
) -> bool
```

Display this object in jupyter.

### `runs`

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/apis/public/reports.py#L149-L170)

```python
runs(
    section, per_page=50, only_selected=(True)
)
```

### `snake_to_camel`

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/apis/attrs.py#L12-L14)

```python
snake_to_camel(
    string
)
```

### `to_html`

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/apis/public/reports.py#L229-L240)

```python
to_html(
    height=1024, hidden=(False)
)
```

Generate HTML containing an iframe displaying this report.
