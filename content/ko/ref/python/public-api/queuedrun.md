---
title: QueuedRun
menu:
  reference:
    identifier: ko-ref-python-public-api-queuedrun
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/jobs.py#L214-L417 >}}

엔티티 및 프로젝트와 연결된 단일 대기열 run입니다. `run = queued_run.wait_until_running()` 또는 `run = queued_run.wait_until_finished()`를 호출하여 run에 액세스합니다.

```python
QueuedRun(
    client, entity, project, queue_name, run_queue_item_id,
    project_queue=LAUNCH_DEFAULT_PROJECT, priority=None
)
```

| 속성 |  |
| :--- | :--- |

## 메소드

### `delete`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/jobs.py#L338-L387)

```python
delete(
    delete_artifacts=(False)
)
```

wandb 백엔드에서 지정된 대기열 run을 삭제합니다.

### `wait_until_finished`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/jobs.py#L328-L336)

```python
wait_until_finished()
```

### `wait_until_running`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/jobs.py#L389-L414)

```python
wait_until_running()
```