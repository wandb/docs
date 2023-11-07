# QueuedRun

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.16.0/wandb/apis/public.py#L2459-L2662' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


A single queued run associated with an entity and project. Call `run = queued_run.wait_until_running()` or `run = queued_run.wait_until_finished()` to access the run.

```python
QueuedRun(
    client, entity, project, queue_name, run_queue_item_id, container_job=(False),
    project_queue=LAUNCH_DEFAULT_PROJECT
)
```

| Attributes |  |
| :--- | :--- |

## Methods

### `delete`

[View source](https://www.github.com/wandb/wandb/tree/v0.16.0/wandb/apis/public.py#L2583-L2632)

```python
delete(
    delete_artifacts=(False)
)
```

Delete the given queued run from the wandb backend.

### `wait_until_finished`

[View source](https://www.github.com/wandb/wandb/tree/v0.16.0/wandb/apis/public.py#L2573-L2581)

```python
wait_until_finished()
```

### `wait_until_running`

[View source](https://www.github.com/wandb/wandb/tree/v0.16.0/wandb/apis/public.py#L2634-L2659)

```python
wait_until_running()
```
