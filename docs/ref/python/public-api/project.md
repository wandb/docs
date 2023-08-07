# Project

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/ab3de55f03aa897601792d5b80d5072d760a6801/wandb/apis/public.py#L1617-L1699' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


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

[View source](https://www.github.com/wandb/wandb/tree/ab3de55f03aa897601792d5b80d5072d760a6801/wandb/apis/public.py#L1650-L1652)

```python
artifacts_types(
    per_page=50
)
```

### `display`

[View source](https://www.github.com/wandb/wandb/tree/ab3de55f03aa897601792d5b80d5072d760a6801/wandb/apis/public.py#L1131-L1142)

```python
display(
    height=420, hidden=(False)
) -> bool
```

Display this object in jupyter.

### `snake_to_camel`

[View source](https://www.github.com/wandb/wandb/tree/ab3de55f03aa897601792d5b80d5072d760a6801/wandb/apis/public.py#L1127-L1129)

```python
snake_to_camel(
    string
)
```

### `sweeps`

[View source](https://www.github.com/wandb/wandb/tree/ab3de55f03aa897601792d5b80d5072d760a6801/wandb/apis/public.py#L1654-L1699)

```python
sweeps()
```

### `to_html`

[View source](https://www.github.com/wandb/wandb/tree/ab3de55f03aa897601792d5b80d5072d760a6801/wandb/apis/public.py#L1634-L1642)

```python
to_html(
    height=420, hidden=(False)
)
```

Generate HTML containing an iframe displaying this project.
