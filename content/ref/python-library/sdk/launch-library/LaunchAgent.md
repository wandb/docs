---
title: LaunchAgent
object_type: launch_apis_namespace
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/sdk/launch/agent/agent.py >}}




## <kbd>class</kbd> `LaunchAgent`
Launch agent class which polls run given run queues and launches runs for wandb launch. 

### <kbd>method</kbd> `LaunchAgent.__init__`

```python
__init__(api: wandb.apis.internal.Api, config: Dict[str, Any])
```

Initialize a launch agent. 



**Arguments:**
 
 - `api`:  Api object to use for making requests to the backend. 
 - `config`:  Config dictionary for the agent. 


---

### <kbd>property</kbd> LaunchAgent.num_running_jobs

Return the number of jobs not including schedulers. 

---

### <kbd>property</kbd> LaunchAgent.num_running_schedulers

Return just the number of schedulers. 

---

### <kbd>property</kbd> LaunchAgent.thread_ids

Returns a list of keys running thread ids for the agent. 



---

### <kbd>method</kbd> `LaunchAgent.check_sweep_state`

```python
check_sweep_state(
    launch_spec: Dict[str, Any],
    api: wandb.apis.internal.Api
) → None
```

Check the state of a sweep before launching a run for the sweep. 

---

### <kbd>method</kbd> `LaunchAgent.fail_run_queue_item`

```python
fail_run_queue_item(
    run_queue_item_id: str,
    message: str,
    phase: str,
    files: Optional[List[str]] = None
) → None
```





---

### <kbd>method</kbd> `LaunchAgent.finish_thread_id`

```python
finish_thread_id(
    thread_id: int,
    exception: Optional[Exception, wandb.sdk.launch.errors.LaunchDockerError] = None
) → None
```

Removes the job from our list for now. 

---

### <kbd>method</kbd> `LaunchAgent.get_job_and_queue`

```python
get_job_and_queue() → Optional[wandb.sdk.launch.agent.agent.JobSpecAndQueue]
```





---

### <kbd>classmethod</kbd> `LaunchAgent.initialized`

```python
initialized() → bool
```

Return whether the agent is initialized. 

---

### <kbd>method</kbd> `LaunchAgent.loop`

```python
loop() → None
```

Loop infinitely to poll for jobs and run them. 



**Raises:**
 
 - `KeyboardInterrupt`:  if the agent is requested to stop. 

---

### <kbd>classmethod</kbd> `LaunchAgent.name`

```python
name() → str
```

Return the name of the agent. 

---

### <kbd>method</kbd> `LaunchAgent.pop_from_queue`

```python
pop_from_queue(queue: str) → Any
```

Pops an item off the runqueue to run as a job. 



**Arguments:**
 
 - `queue`:  Queue to pop from. 



**Returns:**
 Item popped off the queue. 



**Raises:**
 
 - `Exception`:  if there is an error popping from the queue. 

---

### <kbd>method</kbd> `LaunchAgent.print_status`

```python
print_status() → None
```

Prints the current status of the agent. 

---

### <kbd>method</kbd> `LaunchAgent.run_job`

```python
run_job(
    job: Dict[str, Any],
    queue: str,
    file_saver: wandb.sdk.launch.agent.run_queue_item_file_saver.RunQueueItemFileSaver
) → None
```

Set up project and run the job. 



**Arguments:**
 
 - `job`:  Job to run. 

---

### <kbd>method</kbd> `LaunchAgent.task_run_job`

```python
task_run_job(
    launch_spec: Dict[str, Any],
    job: Dict[str, Any],
    default_config: Dict[str, Any],
    api: wandb.apis.internal.Api,
    job_tracker: wandb.sdk.launch.agent.job_status_tracker.JobAndRunStatusTracker
) → None
```





---

### <kbd>method</kbd> `LaunchAgent.update_status`

```python
update_status(status: str) → None
```

Update the status of the agent. 



**Arguments:**
 
 - `status`:  Status to update the agent to. 

