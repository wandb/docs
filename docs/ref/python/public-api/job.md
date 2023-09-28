# Job

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/a8b0374a129d723f15c8e78682dd743ef90f3dfb/wandb/apis/public.py#L4745-L4905' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


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

[View source](https://www.github.com/wandb/wandb/tree/a8b0374a129d723f15c8e78682dd743ef90f3dfb/wandb/apis/public.py#L4865-L4905)

```python
call(
    config, project=None, entity=None, queue=None, resource="local-container",
    resource_args=None, project_queue=None
)
```

### `set_entrypoint`

[View source](https://www.github.com/wandb/wandb/tree/a8b0374a129d723f15c8e78682dd743ef90f3dfb/wandb/apis/public.py#L4862-L4863)

```python
set_entrypoint(
    entrypoint: List[str]
)
```
