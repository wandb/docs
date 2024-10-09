# RunQueue

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/jobs.py#L433-L651' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>


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
|  `items` |  최대 100개까지의 대기열에 등록된 runs. 이 리스트를 수정해도 대기열이나 등록된 항목에는 변경되지 않습니다! |

## 메소드

### `create`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/jobs.py#L638-L651)

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

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/jobs.py#L514-L536)

```python
delete()
```

wandb 백엔드에서 run 대기열을 삭제합니다.