---
title: LaunchAgent
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L164-L924 >}}

Launch agent class which polls run given run queues and launches runs for wandb launch.

```python
LaunchAgent(
    api: Api,
    config: Dict[str, Any]
)
```

| Arguments |  |
| :--- | :--- |
|  `api` |  Api object to use for making requests to the backend. |
|  `config` |  Config dictionary for the agent. |

| Attributes |  |
| :--- | :--- |
|  `num_running_jobs` |  Return the number of jobs not including schedulers. |
|  `num_running_schedulers` |  Return just the number of schedulers. |
|  `thread_ids` |  Returns a list of keys running thread ids for the agent. |

## Methods

### `check_sweep_state`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L786-L803)

```python
check_sweep_state(
    launch_spec, api
)
```

Check the state of a sweep before launching a run for the sweep.

### `fail_run_queue_item`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L295-L304)

```python
fail_run_queue_item(
    run_queue_item_id, message, phase, files=None
)
```

### `finish_thread_id`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L416-L509)

```python
finish_thread_id(
    thread_id, exception=None
)
```

Removes the job from our list for now.

### `get_job_and_queue`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L908-L915)

```python
get_job_and_queue()
```

### `initialized`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L190-L193)

```python
@classmethod
initialized() -> bool
```

Return whether the agent is initialized.

### `loop`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L572-L653)

```python
loop()
```

Loop infinitely to poll for jobs and run them.

| Raises |  |
| :--- | :--- |
|  `KeyboardInterrupt` |  if the agent is requested to stop. |

### `name`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L180-L188)

```python
@classmethod
name() -> str
```

Return the name of the agent.

### `pop_from_queue`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L340-L363)

```python
pop_from_queue(
    queue
)
```

Pops an item off the runqueue to run as a job.

| Arguments |  |
| :--- | :--- |
|  `queue` |  Queue to pop from. |

| Returns |  |
| :--- | :--- |
|  Item popped off the queue. |

| Raises |  |
| :--- | :--- |
|  `Exception` |  if there is an error popping from the queue. |

### `print_status`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L365-L381)

```python
print_status() -> None
```

Prints the current status of the agent.

### `run_job`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L511-L541)

```python
run_job(
    job, queue, file_saver
)
```

Set up project and run the job.

| Arguments |  |
| :--- | :--- |
|  `job` |  Job to run. |

### `task_run_job`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L656-L688)

```python
task_run_job(
    launch_spec, job, default_config, api, job_tracker
)
```

### `update_status`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L383-L394)

```python
update_status(
    status
)
```

Update the status of the agent.

| Arguments |  |
| :--- | :--- |
|  `status` |  Status to update the agent to. |
