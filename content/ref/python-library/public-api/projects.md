---
title: projects
object_type: public_apis_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/apis/public/projects.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API for Projects. 

This module provides classes for interacting with W&B projects and their associated data. Classes include: 

Projects: A paginated collection of projects associated with an entity 
- Iterate through all projects 
- Access project metadata 
- Query project information 

Project: A single project that serves as a namespace for runs 
- Access project properties 
- Work with artifacts and their types 
- Manage sweeps 
- Generate HTML representations for Jupyter 



**Example:**
 ```python
from wandb.apis.public import Api

# Initialize API
api = Api()

# Get all projects for an entity
projects = api.projects("entity")

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
An iterable collection of `Project` objects. 

An iterable interface to access projects created and saved by the entity. 



**Args:**
 
 - `client` (`wandb.apis.internal.Api`):  The API client instance to use. 
 - `entity` (str):  The entity name (username or team) to fetch projects for. 
 - `per_page` (int):  Number of projects to fetch per request (default is 50). 



**Example:**
 ```python
from wandb.apis.public.api import Api

# Initialize the API client
api = Api()

# Find projects that belong to this entity
projects = api.projects(entity="entity")

# Iterate over files
for project in projects:
    print(f"Project: {project.name}")
    print(f"- URL: {project.url}")
    print(f"- Created at: {project.created_at}")
    print(f"- Is benchmark: {project.is_benchmark}")
``` 

### <kbd>method</kbd> `Projects.__init__`

```python
__init__(client, entity, per_page=50)
```






---

### <kbd>property</kbd> Projects.cursor

Returns the cursor position for pagination of project results. 

---

### <kbd>property</kbd> Projects.length

Returns the total number of projects. 

Note: This property is not available for projects. 

---

### <kbd>property</kbd> Projects.more

Returns `True` if there are more projects to fetch. Returns `False` if there are no more projects to fetch. 



---

### <kbd>method</kbd> `Projects.convert_objects`

```python
convert_objects()
```

Converts GraphQL edges to File objects. 


---

## <kbd>class</kbd> `Project`
A project is a namespace for runs. 



**Args:**
 
 - `client`:  W&B API client instance. 
 - `name` (str):  The name of the project. 
 - `entity` (str):  The entity name that owns the project. 

### <kbd>method</kbd> `Project.__init__`

```python
__init__(client, entity, project, attrs)
```






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
sweeps()
```

Fetches all sweeps associated with the project. 

---

### <kbd>method</kbd> `Project.to_html`

```python
to_html(height=420, hidden=False)
```

Generate HTML containing an iframe displaying this project. 


