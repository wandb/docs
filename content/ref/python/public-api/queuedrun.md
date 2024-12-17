---
title: QueuedRun
---

{{< cta-button githubLink="https://www.github.com/wandb/wandb/tree/v0.18.7/wandb/apis/public/jobs.py#L220-L423" >}}

A single queued run associated with an entity and project. Call `run = queued_run.wait_until_running()` or `run = queued_run.wait_until_finished()` to access the run.

```python
QueuedRun(
    client, entity, project, queue_name, run_queue_item_id,
    project_queue=LAUNCH_DEFAULT_PROJECT, priority=None
)
```

| Attributes |  |
| :--- | :--- |

## Methods

### `delete`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.7/wandb/apis/public/jobs.py#L344-L393)

```python
delete(
    delete_artifacts=(False)
)
```

Delete the given queued run from the wandb backend.

### `wait_until_finished`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.7/wandb/apis/public/jobs.py#L334-L342)

```python
wait_until_finished()
```

### `wait_until_running`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.7/wandb/apis/public/jobs.py#L395-L420)

```python
wait_until_running()
```
