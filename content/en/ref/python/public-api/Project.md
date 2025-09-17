---
title: Project
namespace: public_apis_namespace
python_object_type: class
---
{{< readfile file="/_includes/public-api-use.md" >}}


{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/projects.py >}}




## <kbd>class</kbd> `Project`
A project is a namespace for runs.

### <kbd>method</kbd> `Project.__init__`

```python
__init__(
    client: wandb.apis.public.api.RetryingClient,
    entity: str,
    project: str,
    attrs: dict
) → Project
```

**Args:**
 
 - `client`:  W&B API client instance. 
 - `name` (str):  The name of the project. 
 - `entity` (str):  The entity name that owns the project. 


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

