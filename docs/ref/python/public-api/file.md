# File

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.15.9/wandb/apis/public.py#L3085-L3164' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


File is a class associated with a file saved by wandb.

```python
File(
    client, attrs
)
```

| Attributes |  |
| :--- | :--- |

## Methods

### `delete`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.9/wandb/apis/public.py#L3144-L3157)

```python
delete()
```

### `display`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.9/wandb/apis/public.py#L1131-L1142)

```python
display(
    height=420, hidden=(False)
) -> bool
```

Display this object in jupyter.

### `download`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.9/wandb/apis/public.py#L3111-L3142)

```python
download(
    root: str = ".",
    replace: bool = (False),
    exist_ok: bool = (False)
) -> io.TextIOWrapper
```

Downloads a file previously saved by a run from the wandb server.

| Arguments |  |
| :--- | :--- |
|  replace (boolean): If `True`, download will overwrite a local file if it exists. Defaults to `False`. root (str): Local directory to save the file. Defaults to ".". exist_ok (boolean): If `True`, will not raise ValueError if file already exists and will not re-download unless replace=True. Defaults to `False`. |

| Raises |  |
| :--- | :--- |
|  `ValueError` if file already exists, replace=False and exist_ok=False. |

### `snake_to_camel`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.9/wandb/apis/public.py#L1127-L1129)

```python
snake_to_camel(
    string
)
```

### `to_html`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.9/wandb/apis/public.py#L1144-L1145)

```python
to_html(
    *args, **kwargs
)
```
