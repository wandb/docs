---
title: File
---

{{< cta-button githubLink="https://www.github.com/wandb/wandb/tree/v0.18.7/wandb/apis/public/files.py#L109-L210" >}}

File is a class associated with a file saved by wandb.

```python
File(
    client, attrs, run=None
)
```

| Attributes |  |
| :--- | :--- |
|  `path_uri` |  Returns the uri path to the file in the storage bucket. |

## Methods

### `delete`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.1/wandb/apis/public/files.py#L193-L223)

```python
delete()
```

### `display`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.1/wandb/apis/attrs.py#L16-L37)

```python
display(
    height=420, hidden=(False)
) -> bool
```

Display this object in jupyter.

### `download`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.1/wandb/apis/public/files.py#L152-L191)

```python
download(
    root: str = ".",
    replace: bool = (False),
    exist_ok: bool = (False),
    api: Optional[Api] = None
) -> io.TextIOWrapper
```

Downloads a file previously saved by a run from the wandb server.

| Args |  |
| :--- | :--- |
|  replace (boolean): If `True`, download will overwrite a local file if it exists. Defaults to `False`. root (str): Local directory to save the file. Defaults to ".". exist_ok (boolean): If `True`, will not raise ValueError if file already exists and will not re-download unless replace=True. Defaults to `False`. api (Api, optional): If given, the `Api` instance used to download the file. |

| Raises |  |
| :--- | :--- |
|  `ValueError` if file already exists, replace=False and exist_ok=False. |

### `snake_to_camel`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.1/wandb/apis/attrs.py#L12-L14)

```python
snake_to_camel(
    string
)
```

### `to_html`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.1/wandb/apis/attrs.py#L39-L40)

```python
to_html(
    *args, **kwargs
)
```
