---
title: sweeps
object_type: client_type
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/apis/public/sweeps.py >}}




# <kbd>module</kbd> `wandb.apis.public`
Public API: sweeps. 



## <kbd>class</kbd> `Sweep`
A set of runs associated with a sweep. 



**Examples:**
  Instantiate with: ```
     api = wandb.Api()
     sweep = api.sweep(path / to / sweep)
    ``` 



**Attributes:**
 
 - `runs`:  (`Runs`) list of runs 
 - `id`:  (str) sweep id 
 - `project`:  (str) name of project 
 - `config`:  (str) dictionary of sweep configuration 
 - `state`:  (str) the state of the sweep 
 - `expected_run_count`:  (int) number of expected runs for the sweep 

### <kbd>method</kbd> `Sweep.__init__`

```python
__init__(client, entity, project, sweep_id, attrs=None)
```






---

### <kbd>property</kbd> Sweep.config





---

### <kbd>property</kbd> Sweep.entity





---

### <kbd>property</kbd> Sweep.expected_run_count

Return the number of expected runs in the sweep or None for infinite runs. 

---

### <kbd>property</kbd> Sweep.name





---

### <kbd>property</kbd> Sweep.order





---

### <kbd>property</kbd> Sweep.path





---

### <kbd>property</kbd> Sweep.url





---

### <kbd>property</kbd> Sweep.username







---

### <kbd>method</kbd> `Sweep.best_run`

```python
best_run(order=None)
```

Return the best run sorted by the metric defined in config or the order passed in. 

---

### <kbd>classmethod</kbd> `Sweep.get`

```python
get(
    client,
    entity=None,
    project=None,
    sid=None,
    order=None,
    query=None,
    **kwargs
)
```

Execute a query against the cloud backend. 

---

### <kbd>method</kbd> `Sweep.load`

```python
load(force: bool = False)
```





---

### <kbd>method</kbd> `Sweep.to_html`

```python
to_html(height=420, hidden=False)
```

Generate HTML containing an iframe displaying this sweep. 


