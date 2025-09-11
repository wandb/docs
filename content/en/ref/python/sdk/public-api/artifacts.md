---
title: Artifacts
object_type: public_apis_namespace
data_type_classification: module
---
{{< readfile file="/_includes/public-api-use.md" >}}

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/artifacts.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API for Artifact objects. 

This module provides classes for interacting with W&B artifacts and their collections. 


## <kbd>class</kbd> `ArtifactTypes`
An lazy iterator of `ArtifactType` objects for a specific project. 


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

---

### <kbd>property</kbd> ArtifactType.name

The name of the artifact type. 



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


## <kbd>class</kbd> `ArtifactCollections`
Artifact collections of a specific type in a project. 



**Args:**
 
 - `client`:  The client instance to use for querying W&B. 
 - `entity`:  The entity (user or team) that owns the project. 
 - `project`:  The name of the project to query for artifact collections. 
 - `type_name`:  The name of the artifact type for which to fetch collections. 
 - `per_page`:  The number of artifact collections to fetch per page. Default is 50. 


### <kbd>property</kbd> ArtifactCollections.length





---




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

---

### <kbd>property</kbd> ArtifactCollection.created_at

The creation date of the artifact collection. 

---

### <kbd>property</kbd> ArtifactCollection.description

A description of the artifact collection. 

---

### <kbd>property</kbd> ArtifactCollection.id

The unique identifier of the artifact collection. 

---

### <kbd>property</kbd> ArtifactCollection.name

The name of the artifact collection. 

---

### <kbd>property</kbd> ArtifactCollection.tags

The tags associated with the artifact collection. 

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


---

## <kbd>class</kbd> `Artifacts`
An iterable collection of artifact versions associated with a project. 

Optionally pass in filters to narrow down the results based on specific criteria. 



**Args:**
 
 - `client`:  The client instance to use for querying W&B. 
 - `entity`:  The entity (user or team) that owns the project. 
 - `project`:  The name of the project to query for artifacts. 
 - `collection_name`:  The name of the artifact collection to query. 
 - `type`:  The type of the artifacts to query. Common examples include  "dataset" or "model". 
 - `filters`:  Optional mapping of filters to apply to the query. 
 - `order`:  Optional string to specify the order of the results. 
 - `per_page`:  The number of artifact versions to fetch per page. Default is 50. 
 - `tags`:  Optional string or list of strings to filter artifacts by tags. 


### <kbd>property</kbd> Artifacts.length





---



## <kbd>class</kbd> `RunArtifacts`
An iterable collection of artifacts associated with a specific run. 


### <kbd>property</kbd> RunArtifacts.length





---



## <kbd>class</kbd> `ArtifactFiles`
A paginator for files in an artifact. 


### <kbd>property</kbd> ArtifactFiles.length





---


### <kbd>property</kbd> ArtifactFiles.path

Returns the path of the artifact. 



---


