---
title: RunQueue
menu:
  reference:
    identifier: ja-ref-python-public-api-runqueue
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/jobs.py#L427-L653 >}}

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
|  `items` |  最初の100個までのキューに入った run 。このリストを変更しても、キューやキューに入ったアイテムは変更されません！ |

## メソッド

### `create`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/jobs.py#L640-L653)

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

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/jobs.py#L514-L536)

```python
delete()
```

wandb バックエンドから run キューを削除します。
```