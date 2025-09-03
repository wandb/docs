---
title: sweeps
object_type: public_apis_namespace
data_type_classification: module
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

## <kbd>class</kbd> `Sweeps`
A lazy iterator over a collection of `Sweep` objects. 



**Examples:**
 ```python
from wandb.apis.public import Api

sweeps = Api().project(name="project_name", entity="entity").sweeps()

# Iterate over sweeps and print details
for sweep in sweeps:
     print(f"Sweep name: {sweep.name}")
     print(f"Sweep ID: {sweep.id}")
     print(f"Sweep URL: {sweep.url}")
     print("----------")
``` 

### <kbd>method</kbd> `Sweeps.__init__`

```python
__init__(
    client: wandb.apis.public.api.RetryingClient,
    entity: str,
    project: str,
    per_page: int = 50
) → Sweeps
```

An iterable collection of `Sweep` objects. 



**Args:**
 
 - `client`:  The API client used to query W&B. 
 - `entity`:  The entity which owns the sweeps. 
 - `project`:  The project which contains the sweeps. 
 - `per_page`:  The number of sweeps to fetch per request to the API. 


---


### <kbd>property</kbd> Sweeps.length





---




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

Returns the first name that exists in the following priority order: 

1. User-edited display name 2. Name configured at creation time 3. Sweep ID 

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
    client: 'RetryingClient',
    entity: Optional[str] = None,
    project: Optional[str] = None,
    sid: Optional[str] = None,
    order: Optional[str] = None,
    query: Optional[str] = None,
    **kwargs
)
```

Execute a query against the cloud backend. 



**Args:**
 
 - `client`:  The client to use to execute the query. 
 - `entity`:  The entity (username or team) that owns the project. 
 - `project`:  The name of the project to fetch sweep from. 
 - `sid`:  The sweep ID to query. 
 - `order`:  The order in which the sweep's runs are returned. 
 - `query`:  The query to use to execute the query. 
 - `**kwargs`:  Additional keyword arguments to pass to the query. 

---


### <kbd>method</kbd> `Sweep.to_html`

```python
to_html(height=420, hidden=False)
```

Generate HTML containing an iframe displaying this sweep. 


