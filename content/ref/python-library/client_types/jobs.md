---
title: jobs
object_type: client_type
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/apis/public/jobs.py >}}




# <kbd>module</kbd> `wandb.apis.public`
Public API: jobs. 



## <kbd>class</kbd> `Job`




### <kbd>method</kbd> `Job.__init__`

```python
__init__(api: 'Api', name, path: Optional[str] = None) → None
```






---

### <kbd>property</kbd> Job.name







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





---

### <kbd>method</kbd> `Job.set_entrypoint`

```python
set_entrypoint(entrypoint: List[str])
```






---

## <kbd>class</kbd> `QueuedRun`
A single queued run associated with an entity and project. Call `run = queued_run.wait_until_running()` or `run = queued_run.wait_until_finished()` to access the run. 

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





---

### <kbd>property</kbd> QueuedRun.id





---

### <kbd>property</kbd> QueuedRun.project





---

### <kbd>property</kbd> QueuedRun.queue_name





---

### <kbd>property</kbd> QueuedRun.state







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





---

### <kbd>method</kbd> `QueuedRun.wait_until_running`

```python
wait_until_running()
```






---

## <kbd>class</kbd> `RunQueue`




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





---

### <kbd>property</kbd> RunQueue.default_resource_config





---

### <kbd>property</kbd> RunQueue.entity





---

### <kbd>property</kbd> RunQueue.external_links





---

### <kbd>property</kbd> RunQueue.id





---

### <kbd>property</kbd> RunQueue.items

Up to the first 100 queued runs. Modifying this list will not modify the queue or any enqueued items! 

---

### <kbd>property</kbd> RunQueue.name





---

### <kbd>property</kbd> RunQueue.prioritization_mode





---

### <kbd>property</kbd> RunQueue.template_variables





---

### <kbd>property</kbd> RunQueue.type







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





---

### <kbd>method</kbd> `RunQueue.delete`

```python
delete()
```

Delete the run queue from the wandb backend. 


