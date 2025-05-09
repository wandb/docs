---
title: QueuedRun
menu:
  reference:
    identifier: ja-ref-python-public-api-queuedrun
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/jobs.py#L214-L417 >}}

エンティティとプロジェクトに関連付けられた単一のキューに入った run。`run = queued_run.wait_until_running()` または `run = queued_run.wait_until_finished()` を呼び出して run に アクセスします。

```python
QueuedRun(
    client, entity, project, queue_name, run_queue_item_id,
    project_queue=LAUNCH_DEFAULT_PROJECT, priority=None
)
```

| 属性 |  |
| :--- | :--- |

## メソッド

### `delete`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/jobs.py#L338-L387)

```python
delete(
    delete_artifacts=(False)
)
```

指定されたキューに入った run を wandb のバックエンドから削除します。

### `wait_until_finished`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/jobs.py#L328-L336)

```python
wait_until_finished()
```

### `wait_until_running`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/jobs.py#L389-L414)

```python
wait_until_running()
```