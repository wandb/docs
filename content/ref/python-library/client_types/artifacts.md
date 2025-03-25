---
title: artifacts
object_type: client_type
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/apis/public/artifacts.py >}}




# <kbd>module</kbd> `wandb.apis.public`
Public API: artifacts. 



## <kbd>function</kbd> `server_supports_artifact_collections_gql_edges`

```python
server_supports_artifact_collections_gql_edges(
    client: 'RetryingClient',
    warn: bool = False
) → bool
```






---

## <kbd>function</kbd> `artifact_collection_edge_name`

```python
artifact_collection_edge_name(server_supports_artifact_collections: bool) → str
```






---

## <kbd>function</kbd> `artifact_collection_plural_edge_name`

```python
artifact_collection_plural_edge_name(
    server_supports_artifact_collections: bool
) → str
```






---

## <kbd>class</kbd> `ArtifactTypes`




### <kbd>method</kbd> `ArtifactTypes.__init__`

```python
__init__(
    client: wandb_gql.client.Client,
    entity: str,
    project: str,
    per_page: Optional[int] = 50
)
```






---

### <kbd>property</kbd> ArtifactTypes.cursor





---

### <kbd>property</kbd> ArtifactTypes.length





---

### <kbd>property</kbd> ArtifactTypes.more







---

### <kbd>method</kbd> `ArtifactTypes.convert_objects`

```python
convert_objects()
```





---

### <kbd>method</kbd> `ArtifactTypes.update_variables`

```python
update_variables()
```






---

## <kbd>class</kbd> `ArtifactType`




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





---

### <kbd>property</kbd> ArtifactType.name







---

### <kbd>method</kbd> `ArtifactType.collection`

```python
collection(name)
```





---

### <kbd>method</kbd> `ArtifactType.collections`

```python
collections(per_page=50)
```

Artifact collections. 

---

### <kbd>method</kbd> `ArtifactType.load`

```python
load()
```






---

## <kbd>class</kbd> `ArtifactCollections`




### <kbd>method</kbd> `ArtifactCollections.__init__`

```python
__init__(
    client: wandb_gql.client.Client,
    entity: str,
    project: str,
    type_name: str,
    per_page: Optional[int] = 50
)
```






---

### <kbd>property</kbd> ArtifactCollections.cursor





---

### <kbd>property</kbd> ArtifactCollections.length





---

### <kbd>property</kbd> ArtifactCollections.more







---

### <kbd>method</kbd> `ArtifactCollections.convert_objects`

```python
convert_objects()
```





---

### <kbd>method</kbd> `ArtifactCollections.update_variables`

```python
update_variables()
```






---

## <kbd>class</kbd> `ArtifactCollection`




### <kbd>method</kbd> `ArtifactCollection.__init__`

```python
__init__(
    client: wandb_gql.client.Client,
    entity: str,
    project: str,
    name: str,
    type: str,
    attrs: Optional[Mapping[str, Any]] = None
)
```






---

### <kbd>property</kbd> ArtifactCollection.aliases

Artifact Collection Aliases. 

---

### <kbd>property</kbd> ArtifactCollection.description

A description of the artifact collection. 

---

### <kbd>property</kbd> ArtifactCollection.id





---

### <kbd>property</kbd> ArtifactCollection.name

The name of the artifact collection. 

---

### <kbd>property</kbd> ArtifactCollection.tags

The tags associated with the artifact collection. 

---

### <kbd>property</kbd> ArtifactCollection.type

The type of the artifact collection. 



---

### <kbd>method</kbd> `ArtifactCollection.artifacts`

```python
artifacts(per_page=50)
```

Artifacts. 

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





---

### <kbd>method</kbd> `ArtifactCollection.save`

```python
save() → None
```

Persist any changes made to the artifact collection. 


---

## <kbd>class</kbd> `Artifacts`
An iterable collection of artifact versions associated with a project and optional filter. 

This is generally used indirectly via the `Api`.artifact_versions method. 

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





---

### <kbd>property</kbd> Artifacts.length





---

### <kbd>property</kbd> Artifacts.more







---

### <kbd>method</kbd> `Artifacts.convert_objects`

```python
convert_objects()
```






---

## <kbd>class</kbd> `RunArtifacts`




### <kbd>method</kbd> `RunArtifacts.__init__`

```python
__init__(
    client: wandb_gql.client.Client,
    run: 'Run',
    mode='logged',
    per_page: Optional[int] = 50
)
```






---

### <kbd>property</kbd> RunArtifacts.cursor





---

### <kbd>property</kbd> RunArtifacts.length





---

### <kbd>property</kbd> RunArtifacts.more







---

### <kbd>method</kbd> `RunArtifacts.convert_objects`

```python
convert_objects()
```






---

## <kbd>class</kbd> `ArtifactFiles`




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





---

### <kbd>property</kbd> ArtifactFiles.length





---

### <kbd>property</kbd> ArtifactFiles.more





---

### <kbd>property</kbd> ArtifactFiles.path







---

### <kbd>method</kbd> `ArtifactFiles.convert_objects`

```python
convert_objects()
```





---

### <kbd>method</kbd> `ArtifactFiles.update_variables`

```python
update_variables()
```






