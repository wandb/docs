---
title: artifacts
object_type: public_apis_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/apis/public/artifacts.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API for Artifact Management. 

This module provides classes for interacting with W&B artifacts and their collections. Classes include: 



ArtifactTypes: A paginated collection of artifact types in a project 
- List and query artifact types 
- Access type metadata 
- Create new types 

ArtifactCollection: A collection of related artifacts 
- Manage artifact collections 
- Update metadata and descriptions 
- Work with tags and aliases 
- Change collection types 

Artifacts: A paginated collection of artifact versions 
- Filter and query artifacts 
- Access artifact metadata 
- Download artifacts 

ArtifactFiles: A paginated collection of files within an artifact 
- List and query artifact files 
- Access file metadata 
- Download individual files 



## <kbd>function</kbd> `server_supports_artifact_collections_gql_edges`

```python
server_supports_artifact_collections_gql_edges(
    client: 'RetryingClient',
    warn: bool = False
) → bool
```

Check if W&B server supports GraphQL edges for artifact collections. 

<!-- lazydoc-ignore: internal --> 


---

## <kbd>function</kbd> `artifact_collection_edge_name`

```python
artifact_collection_edge_name(server_supports_artifact_collections: bool) → str
```

Return the GraphQL edge name for artifact collections or sequences. 

<!-- lazydoc-ignore: internal --> 


---

## <kbd>function</kbd> `artifact_collection_plural_edge_name`

```python
artifact_collection_plural_edge_name(
    server_supports_artifact_collections: bool
) → str
```

Return the GraphQL edge name for artifact collections or sequences. 

<!-- lazydoc-ignore: internal --> 


---

## <kbd>class</kbd> `ArtifactTypes`
An iterable collection of artifact types associated with a project. 



**Args:**
 
 - `client`:  The client instance to use for querying W&B. 
 - `entity`:  The entity (user or team) that owns the project. 
 - `project`:  The name of the project to query for artifact types. 
 - `per_page`:  The number of artifact types to fetch per page. Default is 50. 

### <kbd>method</kbd> `ArtifactTypes.__init__`

```python
__init__(
    client: wandb_gql.client.Client,
    entity: str,
    project: str,
    per_page: int = 50
)
```






---

### <kbd>property</kbd> ArtifactTypes.cursor

Returns the cursor position for pagination of file results. 

---

### <kbd>property</kbd> ArtifactTypes.length

Returns `None`. 

---

### <kbd>property</kbd> ArtifactTypes.more

Returns `True` if there are more artifacts to fetch. Returns `False` if there are no more files to fetch. 



---

### <kbd>method</kbd> `ArtifactTypes.convert_objects`

```python
convert_objects()
```

Converts GraphQL edges to ArtifactType objects. 

---

### <kbd>method</kbd> `ArtifactTypes.update_variables`

```python
update_variables()
```

Updates the variables dictionary with the cursor. 


---

## <kbd>class</kbd> `ArtifactType`
An artifact object that satisfies query based on the specified type. 



**Args:**
 
 - `client`:  The client instance to use for querying W&B. 
 - `entity`:  The entity (user or team) that owns the project. 
 - `project`:  The name of the project to query for artifact types. 
 - `type_name`:  The name of the artifact type. 
 - `attrs`:  Optional mapping of attributes to initialize the artifact type. If not provided,  the object will load its attributes from W&B upon initialization. 

### <kbd>method</kbd> `ArtifactType.__init__`

```python
__init__(
    client: wandb_gql.client.Client,
    entity: str,
    project: str,
    type_name: str,
    attrs: Optional[Mapping[str, Any]] = None
)
```






---

### <kbd>property</kbd> ArtifactType.id

The unique identifier of the artifact type. 

---

### <kbd>property</kbd> ArtifactType.name

The name of the artifact type. 



---

### <kbd>method</kbd> `ArtifactType.collection`

```python
collection(name)
```

Get a specific artifact collection by name. 



**Args:**
 
 - `name` (str):  The name of the artifact collection to retrieve. 

---

### <kbd>method</kbd> `ArtifactType.collections`

```python
collections(per_page=50)
```

Get all artifact collections associated with this artifact type. 



**Args:**
 
 - `per_page` (int):  The number of artifact collections to fetch per page.  Default is 50. 

---

### <kbd>method</kbd> `ArtifactType.load`

```python
load()
```

Load the artifact type metadata. 


---

## <kbd>class</kbd> `ArtifactCollections`
An iterable collection of artifact collections associated with a project and artifact type. 



**Args:**
 
 - `client`:  The client instance to use for querying W&B. 
 - `entity`:  The entity (user or team) that owns the project. 
 - `project`:  The name of the project to query for artifact collections. 
 - `type_name`:  The name of the artifact type for which to fetch collections. 
 - `per_page`:  The number of artifact collections to fetch per page. Default is 50. 

### <kbd>method</kbd> `ArtifactCollections.__init__`

```python
__init__(
    client: wandb_gql.client.Client,
    entity: str,
    project: str,
    type_name: str,
    per_page: int = 50
)
```






---

### <kbd>property</kbd> ArtifactCollections.cursor

Returns the cursor position for pagination of file results. 

---

### <kbd>property</kbd> ArtifactCollections.length

Returns the number of artifact collections. 

---

### <kbd>property</kbd> ArtifactCollections.more

Returns `True` if there are more artifact collections to fetch. 

Returns `False` if there are no more files to fetch. 



---

### <kbd>method</kbd> `ArtifactCollections.convert_objects`

```python
convert_objects()
```

Converts GraphQL edges to ArtifactCollection objects. 

---

### <kbd>method</kbd> `ArtifactCollections.update_variables`

```python
update_variables()
```

Updates the variables dictionary with the cursor. 


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

### <kbd>method</kbd> `ArtifactCollection.__init__`

```python
__init__(
    client: wandb_gql.client.Client,
    entity: str,
    project: str,
    name: str,
    type: str,
    organization: Optional[str] = None,
    attrs: Optional[Mapping[str, Any]] = None
)
```






---

### <kbd>property</kbd> ArtifactCollection.aliases

The aliases associated with the artifact collection. 

---

### <kbd>property</kbd> ArtifactCollection.created_at

The creation timestamp of the artifact collection. 

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
artifacts(per_page=50)
```

Get all artifact versions associated with this artifact collection. 

---

### <kbd>method</kbd> `ArtifactCollection.change_type`

```python
change_type(new_type: str) → None
```

Deprecated, change type directly with `save` instead. 

---

### <kbd>method</kbd> `ArtifactCollection.delete`

```python
delete()
```

Delete the entire artifact collection. 

---

### <kbd>method</kbd> `ArtifactCollection.is_sequence`

```python
is_sequence() → bool
```

Return whether the artifact collection is a sequence. 

---

### <kbd>method</kbd> `ArtifactCollection.load`

```python
load()
```

Load the artifact collection metadata. 

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

### <kbd>method</kbd> `Artifacts.__init__`

```python
__init__(
    client: wandb_gql.client.Client,
    entity: str,
    project: str,
    collection_name: str,
    type: str,
    filters: Optional[Mapping[str, Any]] = None,
    order: Optional[str] = None,
    per_page: int = 50,
    tags: Optional[str, List[str]] = None
)
```






---

### <kbd>property</kbd> Artifacts.cursor

Returns the cursor position for pagination of file results. 

---

### <kbd>property</kbd> Artifacts.length

Returns the number of artifact versions. 

---

### <kbd>property</kbd> Artifacts.more

Returns `True` if there are more artifact versions to fetch. 



---

### <kbd>method</kbd> `Artifacts.convert_objects`

```python
convert_objects()
```

Converts GraphQL edges to Artifact objects. 


---

## <kbd>class</kbd> `RunArtifacts`
An iterable collection of artifacts associated with a run. 



**Args:**
 
 - `client`:  The client instance to use for querying W&B. 
 - `run`:  The run object to query for artifacts. 
 - `mode`:  The mode of artifacts to fetch, either "logged"  (output artifacts) or "used" (input artifacts). Default is "logged". 
 - `per_page`:  The number of artifacts to fetch per page. Default is 50. 

### <kbd>method</kbd> `RunArtifacts.__init__`

```python
__init__(
    client: wandb_gql.client.Client,
    run: 'Run',
    mode='logged',
    per_page: int = 50
)
```






---

### <kbd>property</kbd> RunArtifacts.cursor

Returns the cursor position for pagination of file results. 

---

### <kbd>property</kbd> RunArtifacts.length

Returns the number of artifacts associated with the run. 

---

### <kbd>property</kbd> RunArtifacts.more

Returns `True` if there are more artifacts to fetch. 



---

### <kbd>method</kbd> `RunArtifacts.convert_objects`

```python
convert_objects()
```

Converts GraphQL edges to Artifact objects. 


---

## <kbd>class</kbd> `ArtifactFiles`
An iterable collection of files associated with an artifact version. 



**Args:**
 
 - `client`:  The client instance to use for querying W&B. 
 - `artifact`:  The artifact object to query for files. 
 - `names`:  Optional sequence of file names to filter the results by. If  `None`, all files will be returned. 
 - `per_page`:  The number of files to fetch per page. Default is 50. 

### <kbd>method</kbd> `ArtifactFiles.__init__`

```python
__init__(
    client: wandb_gql.client.Client,
    artifact: 'wandb.Artifact',
    names: Optional[Sequence[str]] = None,
    per_page: int = 50
)
```






---

### <kbd>property</kbd> ArtifactFiles.cursor

Returns the cursor position for pagination of file results. 

---

### <kbd>property</kbd> ArtifactFiles.length

Returns the number of files in the artifact. 

---

### <kbd>property</kbd> ArtifactFiles.more

Returns `True` if there are more files to fetch. Returns 

---

### <kbd>property</kbd> ArtifactFiles.path

Returns the path of the artifact. 

The path is a list containingthe entity, project name, and artifact name. 



---

### <kbd>method</kbd> `ArtifactFiles.convert_objects`

```python
convert_objects()
```

Converts GraphQL edges to File objects. 

---

### <kbd>method</kbd> `ArtifactFiles.update_variables`

```python
update_variables()
```

Updates the variables dictionary with the cursor and limit. 


