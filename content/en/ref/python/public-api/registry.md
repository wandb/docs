---
title: Registry
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/registries/registry.py#L35-L370 >}}

A single registry in the Registry.

| Attributes |  |
| :--- | :--- |
|  `allow_all_artifact_types` |  Returns whether all artifact types are allowed in the registry. If `True` then artifacts of any type can be added to this registry. If `False` then artifacts are restricted to the types in `artifact_types` for this registry. |
|  `artifact_types` |  Returns the artifact types allowed in the registry. If `allow_all_artifact_types` is `True` then `artifact_types` reflects the types previously saved or currently used in the registry. If `allow_all_artifact_types` is `False` then artifacts are restricted to the types in `artifact_types`. `python import wandb registry = wandb.Api().create_registry() registry.artifact_types.append("model") registry.save() # once saved, the artifact type `model` cannot be removed registry.artifact_types.append("accidentally_added") registry.artifact_types.remove( "accidentally_added" ) # Types can only be removed if it has not been saved yet ` |
|  `created_at` |  Timestamp of when the registry was created. |
|  `description` |  Description of the registry. |
|  `entity` |  Organization entity of the registry. |
|  `full_name` |  Full name of the registry including the `wandb-registry-` prefix. |
|  `name` |  Name of the registry without the `wandb-registry-` prefix. |
|  `organization` |  Organization name of the registry. |
|  `updated_at` |  Timestamp of when the registry was last updated. |
|  `visibility` |  Visibility of the registry. |

## Methods

### `collections`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/registries/registry.py#L186-L192)

```python
collections(
    filter: (dict[str, Any] | None) = None
) -> Collections
```

Returns the collections belonging to the registry.

### `create`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/registries/registry.py#L202-L266)

```python
@classmethod
create(
    client: Client,
    organization: str,
    name: str,
    visibility: Literal['organization', 'restricted'],
    description: (str | None) = None,
    artifact_types: (list[str] | None) = None
)
```

Create a new registry.

The registry name must be unique within the organization.
This function should be called using `api.create_registry()`

| Args |  |
| :--- | :--- |
|  `client` |  The GraphQL client. |
|  `organization` |  The name of the organization. |
|  `name` |  The name of the registry (without the `wandb-registry-` prefix). |
|  `visibility` |  The visibility level ('organization' or 'restricted'). |
|  `description` |  An optional description for the registry. |
|  `artifact_types` |  An optional list of allowed artifact types. |

| Returns |  |
| :--- | :--- |
|  `Registry` |  The newly created Registry object. |

| Raises |  |
| :--- | :--- |
|  `ValueError` |  If a registry with the same name already exists in the organization or if the creation fails. |

### `delete`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/registries/registry.py#L268-L283)

```python
delete() -> None
```

Delete the registry. This is irreversible.

### `load`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/registries/registry.py#L285-L307)

```python
load() -> None
```

Load the registry attributes from the backend to reflect the latest saved state.

### `save`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/registries/registry.py#L309-L366)

```python
save() -> None
```

Save registry attributes to the backend.

### `versions`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/registries/registry.py#L194-L200)

```python
versions(
    filter: (dict[str, Any] | None) = None
) -> Versions
```

Returns the versions belonging to the registry.
