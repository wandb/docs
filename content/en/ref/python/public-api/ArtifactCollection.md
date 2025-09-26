---
title: ArtifactCollection
namespace: public_apis_namespace
python_object_type: class
---
{{< readfile file="/_includes/public-api-use.md" >}}


{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/artifacts.py >}}




## <kbd>class</kbd> `ArtifactCollection`
An artifact collection that represents a group of related artifacts. 



**Args:**
 
 - `client`:  The client instance to use for querying W&B. 
 - `entity`:  The entity (user or team) that owns the project. 
 - `project`:  The name of the project to query for artifact collections. 
 - `name`:  The name of the artifact collection. 
 - `type`:  The type of the artifact collection (e.g., "dataset", "model"). 
 - `organization`:  Optional organization name if applicable. 
 - `attrs`:  Optional mapping of attributes to initialize the artifact collection.  If not provided, the object will load its attributes from W&B upon  initialization. 


### <kbd>property</kbd> ArtifactCollection.aliases

Artifact Collection Aliases. 



**Returns:**
 - `list[str]`: The aliases property value.
---

### <kbd>property</kbd> ArtifactCollection.created_at

The creation date of the artifact collection. 



**Returns:**
 - `str`: The created_at property value.
---

### <kbd>property</kbd> ArtifactCollection.description

A description of the artifact collection. 



**Returns:**
 - `str`: The description property value.
---

### <kbd>property</kbd> ArtifactCollection.id

The unique identifier of the artifact collection. 



**Returns:**
 - `str`: The id property value.
---

### <kbd>property</kbd> ArtifactCollection.name

The name of the artifact collection. 



**Returns:**
 - `str`: The name property value.
---

### <kbd>property</kbd> ArtifactCollection.tags

The tags associated with the artifact collection. 



**Returns:**
 - `list[str]`: The tags property value.
---

### <kbd>property</kbd> ArtifactCollection.type

Returns the type of the artifact collection. 



---

### <kbd>method</kbd> `ArtifactCollection.artifacts`

```python
artifacts(per_page: 'int' = 50) → Artifacts
```

Get all artifacts in the collection. 

---

### <kbd>method</kbd> `ArtifactCollection.change_type`

```python
change_type(new_type: 'str') → None
```

Deprecated, change type directly with `save` instead. 

---

### <kbd>method</kbd> `ArtifactCollection.delete`

```python
delete() → None
```

Delete the entire artifact collection. 

---

### <kbd>method</kbd> `ArtifactCollection.is_sequence`

```python
is_sequence() → bool
```

Return whether the artifact collection is a sequence. 

---


### <kbd>method</kbd> `ArtifactCollection.save`

```python
save() → None
```

Persist any changes made to the artifact collection. 

