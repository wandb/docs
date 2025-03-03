---
menu:
  default:
    identifier: search_registry
    parent: registry
title: Find registry items 
weight: 7
--- 

Use the global search bar in the W&B Registry App or Python SDK to find a specific registry, collection, collection tag, artifact version name, artifact tag, or artifact alias.

Only items that you have permission to view appear in the search results.

## Interactively search for registry items

To search globally:

1. Navigate to the W&B Registry App.
2. Specify the search term in the search bar at the top of the page. Press Enter to search.

Search results appear below the search bar if the term you specify matches an existing registry, collection name, artifact version tag, collection tag, or alias.

{{< img src="/images/registry/search_registry.gif" alt="" >}}


## Programmatically search for registry items

Use the [`wandb.Api().registries()`]({{< relref "/ref/python/public-api/api.md#registries" >}}) method to filter registries, collections, and artifact versions based on one or more [MongoDB-style filters](https://www.mongodb.com/docs/compass/current/query/filter/). Common filters for registries, collections, and artifact versions include: `name`, `tag`, `created_at`, `updated_at`, and `alias`. For a full list of filter options, see the `filter` argument in the [`wandb.Api().registries()`]({{< relref "/ref/python/public-api/api.md#registries" >}}) API documentation.

To use the `wandb.Api().registries()` method, first import the W&B Python SDK ([`wandb`]({{< relref "/ref/python/_index.md" >}})) library:
```python
import wandb

# (Optional) Create an instance of the
# wandb.Api() class for readability
api = wandb.Api()
```
The proceeding code examples demonstrate some common search scenarios:


```python
# Filter all registries that contain the string `model`
registry_filters = {
    "name": {"$regex": "model"}
}

# Returns an iterable of all registries that
# match the filters
registries = api.registries(filter=registry_filters)
```

```python
# Filter all collections, independent of registry, that 
# contains the string `yolo` in the collection name
collection_filters = {
    "name": {"$regex": "yolo"}
}

# Returns an iterable of all collections that
# match the filters
collections = api.registries().collections(filter=collection_filters)
```

```python
# Filter all collections, independent of registry, that
# contains the string `yolo` in the collection name and
# possesses `cnn` as a tag
collection_filters = {
    "name": {"$regex": "yolo"},
    "tag": "cnn"
}

# Returns an iterable of all collections that
# match the filters
collections = api.registries().collections(filter=collection_filters)
```

```python
# Find all artifact versions that contains the
# string `model` and has either the 
# tag `image-classification` or an `latest` alias
registry_filters = {
    "name": {"$regex": "model"}
}
version_filters = {
    "$or": [
        {"tag": "image-classification"},
        {"alias": "production"}
    ]
}

# Returns an iterable of all artifact versions that
# match the filters
artifacts = api.registries(filter=registry_filters).versions(filter=version_filters)
```
