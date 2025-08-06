---
data_type_classification: module
menu:
  reference:
    identifier: ko-ref-python-public-api-sweeps
object_type: public_apis_namespace
title: sweeps
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/sweeps.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API for Sweeps. 

This module provides classes for interacting with W&B hyperparameter optimization sweeps. 



**Example:**
 ```python
from wandb.apis.public import Api

# Get a specific sweep
sweep = Api().sweep("entity/project/sweep_id")

# Access sweep properties
print(f"Sweep: {sweep.name}")
print(f"State: {sweep.state}")
print(f"Best Loss: {sweep.best_loss}")

# Get best performing run
best_run = sweep.best_run()
print(f"Best Run: {best_run.name}")
print(f"Metrics: {best_run.summary}")
``` 



**Note:**

> This module is part of the W&B Public API and provides read-only access to sweep data. For creating and controlling sweeps, use the wandb.sweep() and wandb.agent() functions from the main wandb package. 

## <kbd>class</kbd> `Sweep`
The set of runs associated with the sweep. 



**Attributes:**
 
 - `runs` (Runs):  List of runs 
 - `id` (str):  Sweep ID 
 - `project` (str):  The name of the project the sweep belongs to 
 - `config` (dict):  Dictionary containing the sweep configuration 
 - `state` (str):  The state of the sweep. Can be "Finished", "Failed",  "Crashed", or "Running". 
 - `expected_run_count` (int):  The number of expected runs for the sweep 

### <kbd>method</kbd> `Sweep.__init__`

```python
__init__(client, entity, project, sweep_id, attrs=None)
```






---

### <kbd>property</kbd> Sweep.config

The sweep configuration used for the sweep. 

---

### <kbd>property</kbd> Sweep.entity

The entity associated with the sweep. 

---

### <kbd>property</kbd> Sweep.expected_run_count

Return the number of expected runs in the sweep or None for infinite runs. 

---

### <kbd>property</kbd> Sweep.name

The name of the sweep. 

If the sweep has a name, it will be returned. Otherwise, the sweep ID will be returned. 

---

### <kbd>property</kbd> Sweep.order

Return the order key for the sweep. 

---

### <kbd>property</kbd> Sweep.path

Returns the path of the project. 

The path is a list containing the entity, project name, and sweep ID. 

---

### <kbd>property</kbd> Sweep.url

The URL of the sweep. 

The sweep URL is generated from the entity, project, the term "sweeps", and the sweep ID.run_id. For SaaS users, it takes the form of `https://wandb.ai/entity/project/sweeps/sweeps_ID`. 

---

### <kbd>property</kbd> Sweep.username

Deprecated. Use `Sweep.entity` instead. 



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


### <kbd>method</kbd> `Sweep.to_html`

```python
to_html(height=420, hidden=False)
```

Generate HTML containing an iframe displaying this sweep.