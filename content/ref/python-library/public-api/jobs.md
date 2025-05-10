---
title: jobs
object_type: public_apis_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/apis/public/jobs.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API for Job Management and Queuing. 

This module provides classes for managing W&B jobs, queued runs, and run queues. Classes include: 

Job: Manage W&B job definitions and execution 
- Load and configure jobs from artifacts 
- Set entrypoints and runtime configurations 
- Execute jobs with different resource types 
- Handle notebook and container-based jobs 

QueuedRun: Track and manage individual queued runs 
- Monitor run state and execution 
- Wait for run completion 
- Access run results and artifacts 
- Delete queued runs 

RunQueue: Manage job queues and execution resources 
- Create and configure run queues 
- Set resource types and configurations 
- Monitor queue items and status 
- Control queue access and priorities 



## <kbd>class</kbd> `Job`




### <kbd>method</kbd> `Job.__init__`

```python
__init__(api: 'Api', name, path: Optional[str] = None) → None
```






---

### <kbd>property</kbd> Job.name

The name of the job. 



---

### <kbd>method</kbd> `Job.call`

```python
call(
    config,
    project=None,
    entity=None,
    queue=None,
    resource='local-container',
    resource_args=None,
    template_variables=None,
    project_queue=None,
    priority=None
)
```

Call the job with the given configuration. 



**Args:**
 
 - `config` (dict):  The configuration to pass to the job.  This should be a dictionary containing key-value pairs that  match the input types defined in the job. 
 - `project` (str, optional):  The project to log the run to. Defaults  to the job's project. 
 - `entity` (str, optional):  The entity to log the run under. Defaults  to the job's entity. 
 - `queue` (str, optional):  The name of the queue to enqueue the job to.  Defaults to None. 
 - `resource` (str, optional):  The resource type to use for execution.  Defaults to "local-container". 
 - `resource_args` (dict, optional):  Additional arguments for the  resource type. Defaults to None. 
 - `template_variables` (dict, optional):  Template variables to use for  the job. Defaults to None. 
 - `project_queue` (str, optional):  The project that manages the queue.  Defaults to None. 
 - `priority` (int, optional):  The priority of the queued run.  Defaults to None. 

---

### <kbd>method</kbd> `Job.set_entrypoint`

```python
set_entrypoint(entrypoint: List[str])
```

Set the entrypoint for the job. 


---

## <kbd>class</kbd> `QueuedRun`
A single queued run associated with an entity and project. 



**Args:**
 
 - `entity`:  The entity associated with the queued run. 
 - `project` (str):  The project where runs executed by the queue are logged to. 
 - `queue_name` (str):  The name of the queue. 
 - `run_queue_item_id` (int):  The id of the run queue item. 
 - `project_queue` (str):  The project that manages the queue. 
 - `priority` (str):  The priority of the queued run. 

Call `run = queued_run.wait_until_running()` or `run = queued_run.wait_until_finished()` to access the run. 

### <kbd>method</kbd> `QueuedRun.__init__`

```python
__init__(
    client,
    entity,
    project,
    queue_name,
    run_queue_item_id,
    project_queue='model-registry',
    priority=None
)
```






---

### <kbd>property</kbd> QueuedRun.entity

The entity associated with the queued run. 

---

### <kbd>property</kbd> QueuedRun.id

The id of the queued run. 

---

### <kbd>property</kbd> QueuedRun.project

The project associated with the queued run. 

---

### <kbd>property</kbd> QueuedRun.queue_name

The name of the queue. 

---

### <kbd>property</kbd> QueuedRun.state

The state of the queued run. 



---

### <kbd>method</kbd> `QueuedRun.delete`

```python
delete(delete_artifacts=False)
```

Delete the given queued run from the wandb backend. 

---

### <kbd>method</kbd> `QueuedRun.wait_until_finished`

```python
wait_until_finished()
```

Wait for the queued run to complete and return the finished run. 

---

### <kbd>method</kbd> `QueuedRun.wait_until_running`

```python
wait_until_running()
```

Wait until the queued run is running and return the run. 


---

## <kbd>class</kbd> `RunQueue`
Class that represents a run queue in W&B. 



**Args:**
 
 - `client`:  W&B API client instance. 
 - `name`:  Name of the run queue 
 - `entity`:  The entity (user or team) that owns this queue 
 - `prioritization_mode`:  Queue priority mode  Can be "DISABLED" or "V0". Defaults to `None`. 
 - `_access`:  Access level for the queue  Can be "project" or "user". Defaults to `None`. 
 - `_default_resource_config_id`:  ID of default resource config 
 - `_default_resource_config`:  Default resource configuration 

### <kbd>method</kbd> `RunQueue.__init__`

```python
__init__(
    client: 'RetryingClient',
    name: str,
    entity: str,
    prioritization_mode: Optional[Literal['DISABLED', 'V0']] = None,
    _access: Optional[Literal['project', 'user']] = None,
    _default_resource_config_id: Optional[int] = None,
    _default_resource_config: Optional[dict] = None
) → None
```






---

### <kbd>property</kbd> RunQueue.access

The access level of the queue. 

---

### <kbd>property</kbd> RunQueue.default_resource_config

The default configuration for resources. 

---

### <kbd>property</kbd> RunQueue.entity

The entity that owns the queue. 

---

### <kbd>property</kbd> RunQueue.external_links

External resource links for the queue. 

---

### <kbd>property</kbd> RunQueue.id

The id of the queue. 

---

### <kbd>property</kbd> RunQueue.items

Up to the first 100 queued runs. Modifying this list will not modify the queue or any enqueued items! 

---

### <kbd>property</kbd> RunQueue.name

The name of the queue. 

---

### <kbd>property</kbd> RunQueue.prioritization_mode

The prioritization mode of the queue. 

Can be set to "DISABLED" or "V0". 

---

### <kbd>property</kbd> RunQueue.template_variables

Variables for resource templates. 

---

### <kbd>property</kbd> RunQueue.type

The resource type for execution. 



---

### <kbd>classmethod</kbd> `RunQueue.create`

```python
create(
    name: str,
    resource: 'RunQueueResourceType',
    entity: Optional[str] = None,
    prioritization_mode: Optional[ForwardRef('RunQueuePrioritizationMode')] = None,
    config: Optional[dict] = None,
    template_variables: Optional[dict] = None
) → RunQueue
```

Create a RunQueue. 



**Args:**
 
 - `name`:  The name of the run queue to create. 
 - `resource`:  The resource type for execution. 
 - `entity`:  The entity (user or team) that will own the queue.  Defaults to the default entity of the API client. 
 - `prioritization_mode`:  The prioritization mode for the queue.  Can be "DISABLED" or "V0". Defaults to None. 
 - `config`:  Optional dictionary for the default resource  configuration. Defaults to None. 
 - `template_variables`:  Optional dictionary for template variables  used in the resource configuration. 

---

### <kbd>method</kbd> `RunQueue.delete`

```python
delete()
```

Delete the run queue from the wandb backend. 


