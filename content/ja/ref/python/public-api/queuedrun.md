---
title: QueuedRun
menu:
  reference:
    identifier: ja-ref-python-public-api-queuedrun
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/jobs.py#L214-L417 >}}

Entity と Project に関連付けられた、単一のキューに登録された run 。run にアクセスするには、`run = queued_run.wait_until_running()` または `run = queued_run.wait_until_finished()` を呼び出します。

```python
QueuedRun(
    client, entity, project, queue_name, run_queue_item_id,
    project_queue=LAUNCH_DEFAULT_PROJECT, priority=None
)
```

| Attributes |  |
| :--- | :--- |

## メソッド

### `delete`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/jobs.py#L338-L387)

```python
delete(
    delete_artifacts=(False)
)
```

指定されたキューに登録された run を wandb バックエンドから削除します。

### `wait_until_finished`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/jobs.py#L328-L336)

```python
wait_until_finished()
```

### `wait_until_running`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/jobs.py#L389-L414)

```python
wait_until_running()
```