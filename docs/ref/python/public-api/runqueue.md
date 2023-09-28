# RunQueue

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/a8b0374a129d723f15c8e78682dd743ef90f3dfb/wandb/apis/public.py#L2666-L2861' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


```python
RunQueue(
    client: RetryingClient,
    name: str,
    entity: str,
    _access: Optional[RunQueueAccessType] = None,
    _default_resource_config_id: Optional[int] = None,
    _default_resource_config: Optional[dict] = None
) -> None
```

| Attributes |  |
| :--- | :--- |
|  `items` |  Up to the first 100 queued runs. Modifying this list will not modify the queue or any enqueued items! |

## Methods

### `create`

[View source](https://www.github.com/wandb/wandb/tree/a8b0374a129d723f15c8e78682dd743ef90f3dfb/wandb/apis/public.py#L2845-L2861)

```python
@classmethod
create(
    name: str,
    resource: "RunQueueResourceType",
    access: "RunQueueAccessType",
    entity: Optional[str] = None,
    config: Optional[dict] = None
) -> "RunQueue"
```

### `delete`

[View source](https://www.github.com/wandb/wandb/tree/a8b0374a129d723f15c8e78682dd743ef90f3dfb/wandb/apis/public.py#L2730-L2752)

```python
delete()
```

Delete the run queue from the wandb backend.
