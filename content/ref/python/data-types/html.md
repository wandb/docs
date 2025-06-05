---
title: Html
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/data_types/html.py#L19-L134 >}}

A class for logging HTML content to W&B.

```python
Html(
    data: Union[str, pathlib.Path, 'TextIO'],
    inject: bool = (True),
    data_is_not_path: bool = (False)
) -> None
```

| Args |  |
| :--- | :--- |
|  `data` |  A string that is a path to a file with the extension ".html", or a string or IO object containing literal HTML. |
|  `inject` |  Add a stylesheet to the HTML object. If set to False the HTML will pass through unchanged. |
|  `data_is_not_path` |  If set to False, the data will be treated as a path to a file. |

## Methods

### `inject_head`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/data_types/html.py#L86-L101)

```python
inject_head() -> None
```
