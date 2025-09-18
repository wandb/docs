---
title: Registry
namespace: public_apis_namespace
python_object_type: class
---
{{< readfile file="/_includes/public-api-use.md" >}}


{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/registries/registry.py >}}




## <kbd>class</kbd> `Registry`
A single registry in the Registry. 

### <kbd>method</kbd> `Registry.__init__`

```python
__init__(
    client: 'Client',
    organization: str,
    entity: str,
    name: str,
    attrs: Optional[Dict[str, Any]] = None
)
```






---

### <kbd>property</kbd> Registry.allow_all_artifact_types

Returns whether all artifact types are allowed in the registry. 

If `True` then artifacts of any type can be added to this registry. If `False` then artifacts are restricted to the types in `artifact_types` for this registry. 

---

### <kbd>property</kbd> Registry.artifact_types

Returns the artifact types allowed in the registry. 

If `allow_all_artifact_types` is `True` then `artifact_types` reflects the types previously saved or currently used in the registry. If `allow_all_artifact_types` is `False` then artifacts are restricted to the types in `artifact_types`. 



**Note:**

> Previously saved artifact types cannot be removed. 
>

**Example:**
 ```python
     registry.artifact_types.append("model")
     registry.save()  # once saved, the artifact type `model` cannot be removed
     registry.artifact_types.append("accidentally_added")
     registry.artifact_types.remove(
         "accidentally_added"
     )  # Types can only be removed if it has not been saved yet
    ``` 



**Returns:**
 - `AddOnlyArtifactTypesList`: The artifact_types property value.
---

### <kbd>property</kbd> Registry.created_at

Timestamp of when the registry was created. 



**Returns:**
 - `str`: The created_at property value.
---

### <kbd>property</kbd> Registry.description

Description of the registry. 



**Returns:**
 - `str`: The description property value.
---

### <kbd>property</kbd> Registry.entity

Organization entity of the registry. 



**Returns:**
 - `str`: The entity property value.
---

### <kbd>property</kbd> Registry.full_name

Full name of the registry including the `wandb-registry-` prefix. 



**Returns:**
 - `str`: The full_name property value.
---

### <kbd>property</kbd> Registry.name

Name of the registry without the `wandb-registry-` prefix. 



**Returns:**
 - `str`: The name property value.
---

### <kbd>property</kbd> Registry.organization

Organization name of the registry. 



**Returns:**
 - `str`: The organization property value.
---

### <kbd>property</kbd> Registry.path





---

### <kbd>property</kbd> Registry.updated_at

Timestamp of when the registry was last updated. 



**Returns:**
 - `str`: The updated_at property value.
---

### <kbd>property</kbd> Registry.visibility

Visibility of the registry. 



**Returns:**
 
 - `Literal["organization", "restricted"]`:  The visibility level. 
        - "organization": Anyone in the organization can view this registry.  You can edit their roles later from the settings in the UI. 
        - "restricted": Only invited members via the UI can access this registry.  Public sharing is disabled. 





**Returns:**
 - `Literal`: The visibility property value.
---

### <kbd>method</kbd> `Registry.collections`

```python
collections(filter: Optional[Dict[str, Any]] = None) → Collections
```

Returns the collections belonging to the registry. 

---

### <kbd>classmethod</kbd> `Registry.create`

```python
create(
    client: 'Client',
    organization: str,
    name: str,
    visibility: Literal['organization', 'restricted'],
    description: Optional[str] = None,
    artifact_types: Optional[List[str]] = None
)
```

Create a new registry. 

The registry name must be unique within the organization. This function should be called using `api.create_registry()` 



**Args:**
 
 - `client`:  The GraphQL client. 
 - `organization`:  The name of the organization. 
 - `name`:  The name of the registry (without the `wandb-registry-` prefix). 
 - `visibility`:  The visibility level ('organization' or 'restricted'). 
 - `description`:  An optional description for the registry. 
 - `artifact_types`:  An optional list of allowed artifact types. 



**Returns:**
 
 - `Registry`:  The newly created Registry object. 



**Raises:**
 
 - `ValueError`:  If a registry with the same name already exists in the  organization or if the creation fails. 

---

### <kbd>method</kbd> `Registry.delete`

```python
delete() → None
```

Delete the registry. This is irreversible. 

---

### <kbd>method</kbd> `Registry.load`

```python
load() → None
```

Load the registry attributes from the backend to reflect the latest saved state. 

---

### <kbd>method</kbd> `Registry.save`

```python
save() → None
```

Save registry attributes to the backend. 

---

### <kbd>method</kbd> `Registry.versions`

```python
versions(filter: Optional[Dict[str, Any]] = None) → Versions
```

Returns the versions belonging to the registry. 

