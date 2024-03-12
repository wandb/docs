
# RunQueue

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/public/jobs.py#L415-L633' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>


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

| 속성 |  |
| :--- | :--- |
|  `items` |  첫 100개의 대기중인 run까지. 이 리스트를 수정해도 대기열이나 대기중인 항목들이 수정되지 않습니다! |

## 메소드

### `create`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/public/jobs.py#L620-L633)

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

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/apis/public/jobs.py#L496-L518)

```python
delete()
```

wandb 백엔드에서 run queue를 삭제합니다.