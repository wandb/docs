
# RunQueue

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/jobs.py#L428-L646' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHubでソースを見る</a></button></p>

```python
RunQueue(
    client: "RetryingClient",
    name: str,
    entity: str,
    prioritization_mode: Optional[RunQueuePrioritizationMode] = None,
    _access: Optional[RunQueueAccessType] = None,
    _default_resource_config_id: Optional[int] = None,
    _default_resource_config: Optional[dict] = None
) -> None
```

| 属性 | 説明 |
| :--- | :--- |
|  `items` |  最初の100件までのキューに入ったrun。 このリストを変更しても、キューやエンキューされた項目には影響しません！ |

## メソッド

### `create`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/jobs.py#L633-L646)

```python
@classmethod
create(
    name: str,
    resource: "RunQueueResourceType",
    entity: Optional[str] = None,
    prioritization_mode: Optional['RunQueuePrioritizationMode'] = None,
    config: Optional[dict] = None,
    template_variables: Optional[dict] = None
) -> "RunQueue"
```

### `delete`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/jobs.py#L509-L531)

```python
delete()
```

wandbのバックエンドからrun queueを削除します。