---
title: RunQueue
---

{{< cta-button githubLink="https://www.github.com/wandb/wandb/tree/v0.18.7/wandb/apis/public/jobs.py#L433-L659" >}}

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

| Attributes |  |
| :--- | :--- |
|  `items` |  Up to the first 100 queued runs. Modifying this list will not modify the queue or any enqueued items. |

## Methods

### `create`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.1/wandb/apis/public/jobs.py#L640-L653)

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

[View source](https://www.github.com/wandb/wandb/tree/v0.19.1/wandb/apis/public/jobs.py#L514-L536)

```python
delete()
```

Delete the run queue from the wandb backend.
