# Job

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.15.12/wandb/apis/public.py#L4750-L4910' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


```python
Job(
    api: Api,
    name,
    path: Optional[str] = None
) -> None
```

| Attributes |  |
| :--- | :--- |

## Methods

### `call`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.12/wandb/apis/public.py#L4870-L4910)

```python
call(
    config, project=None, entity=None, queue=None, resource="local-container",
    resource_args=None, project_queue=None
)
```

### `set_entrypoint`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.12/wandb/apis/public.py#L4867-L4868)

```python
set_entrypoint(
    entrypoint: List[str]
)
```
