---
data_type_classification: module
menu:
  reference:
    identifier: ja-ref-python-public-api-projects
object_type: public_apis_namespace
title: projects
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/projects.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API for Project objects. 

This module provides classes for interacting with W&B projects and their associated data. 



**Example:**
 ```python
from wandb.apis.public import Api

# Get all projects for an entity
projects = Api().projects("entity")

# Access project data
for project in projects:
     print(f"Project: {project.name}")
     print(f"URL: {project.url}")

     # Get artifact types
     for artifact_type in project.artifacts_types():
         print(f"Artifact Type: {artifact_type.name}")

     # Get sweeps
     for sweep in project.sweeps():
         print(f"Sweep ID: {sweep.id}")
         print(f"State: {sweep.state}")
``` 



**Note:**

> This module is part of the W&B Public API and provides methods to access and manage projects. For creating new projects, use wandb.init() with a new project name. 

## <kbd>class</kbd> `Projects`
An lazy iterator of `Project` objects. 

An iterable interface to access projects created and saved by the entity. 



**Args:**
 
 - `client` (`wandb.apis.internal.Api`):  The API client instance to use. 
 - `entity` (str):  The entity name (username or team) to fetch projects for. 
 - `per_page` (int):  Number of projects to fetch per request (default is 50). 



**Example:**
 ```python
from wandb.apis.public.api import Api

# Find projects that belong to this entity
projects = Api().projects(entity="entity")

# Iterate over files
for project in projects:
    print(f"Project: {project.name}")
    print(f"- URL: {project.url}")
    print(f"- Created at: {project.created_at}")
    print(f"- Is benchmark: {project.is_benchmark}")
``` 

### <kbd>method</kbd> `Projects.__init__`

```python
__init__(
    client: wandb.apis.public.api.RetryingClient,
    entity: str,
    per_page: int = 50
) → Projects
```

An iterable collection of `Project` objects. 



**Args:**
 
 - `client`:  The API client used to query W&B. 
 - `entity`:  The entity which owns the projects. 
 - `per_page`:  The number of projects to fetch per request to the API. 


---





## <kbd>class</kbd> `Project`
A project is a namespace for runs. 



**Args:**
 
 - `client`:  W&B API client instance. 
 - `name` (str):  The name of the project. 
 - `entity` (str):  The entity name that owns the project. 

### <kbd>method</kbd> `Project.__init__`

```python
__init__(
    client: wandb.apis.public.api.RetryingClient,
    entity: str,
    project: str,
    attrs: dict
) → Project
```

A single project associated with an entity. 



**Args:**
 
 - `client`:  The API client used to query W&B. 
 - `entity`:  The entity which owns the project. 
 - `project`:  The name of the project to query. 
 - `attrs`:  The attributes of the project. 


---

### <kbd>property</kbd> Project.id





---

### <kbd>property</kbd> Project.path

Returns the path of the project. The path is a list containing the entity and project name. 

---

### <kbd>property</kbd> Project.url

Returns the URL of the project. 



---

### <kbd>method</kbd> `Project.artifacts_types`

```python
artifacts_types(per_page=50)
```

Returns all artifact types associated with this project. 

---

### <kbd>method</kbd> `Project.sweeps`

```python
sweeps(per_page=50)
```

Return a paginated collection of sweeps in this project. 



**Args:**
 
 - `per_page`:  The number of sweeps to fetch per request to the API. 



**Returns:**
 A `Sweeps` object, which is an iterable collection of `Sweep` objects. 

---