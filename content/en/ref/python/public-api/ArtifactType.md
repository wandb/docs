---
title: ArtifactType
namespace: public_apis_namespace
python_object_type: class
---
{{< readfile file="/_includes/public-api-use.md" >}}


{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/artifacts.py >}}




## <kbd>class</kbd> `ArtifactType`
An artifact object that satisfies query based on the specified type. 



**Args:**
 
 - `client`:  The client instance to use for querying W&B. 
 - `entity`:  The entity (user or team) that owns the project. 
 - `project`:  The name of the project to query for artifact types. 
 - `type_name`:  The name of the artifact type. 
 - `attrs`:  Optional mapping of attributes to initialize the artifact type. If not provided,  the object will load its attributes from W&B upon initialization. 


### <kbd>property</kbd> ArtifactType.id

The unique identifier of the artifact type. 



**Returns:**
 - `str`: The id property value.
---

### <kbd>property</kbd> ArtifactType.name

The name of the artifact type. 





**Returns:**
 - `str`: The name property value.
---

### <kbd>method</kbd> `ArtifactType.collection`

```python
collection(name: 'str') → ArtifactCollection
```

Get a specific artifact collection by name. 



**Args:**
 
 - `name` (str):  The name of the artifact collection to retrieve. 

---

### <kbd>method</kbd> `ArtifactType.collections`

```python
collections(per_page: 'int' = 50) → ArtifactCollections
```

Get all artifact collections associated with this artifact type. 



**Args:**
 
 - `per_page` (int):  The number of artifact collections to fetch per page.  Default is 50. 

---

