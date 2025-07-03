---
title: Registry
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/apis/public/registries/registry.py#L32-L358 >}}

A single registry in the Registry.

| Attributes |  |
| :--- | :--- |
|  `allow_all_artifact_types` |  Returns whether all artifact types are allowed in the registry. If `True` then artifacts of any type can be added to this registry. If `False` then artifacts are restricted to the types in `artifact_types` for this registry. |
|  `artifact_types` |  Returns the artifact types allowed in the registry. If `allow_all_artifact_types` is `True` then `artifact_types` reflects the types previously saved or currently used in the registry. If `allow_all_artifact_types` is `False` then artifacts are restricted to the types in `artifact_types`. |
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

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/apis/public/registries/registry.py#L180-L185)

```python
collections(
    filter: Optional[Dict[str, Any]] = None
) -> Collections
```

Returns the collections belonging to the registry.

### `create`

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/apis/public/registries/registry.py#L194-L257)

```python
@classmethod
create(
    client: "Client",
    organization: str,
    name: str,
    visibility: Literal['organization', 'restricted'],
    description: Optional[str] = None,
    artifact_types: Optional[List[str]] = None
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

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/apis/public/registries/registry.py#L259-L273)

```python
delete() -> None
```

Delete the registry. This is irreversible.

### `load`

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/apis/public/registries/registry.py#L275-L296)

```python
load() -> None
```

Load the registry attributes from the backend to reflect the latest saved state.

### `save`

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/apis/public/registries/registry.py#L298-L354)

```python
save() -> None
```

Save registry attributes to the backend.

### `versions`

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/apis/public/registries/registry.py#L187-L192)

```python
versions(
    filter: Optional[Dict[str, Any]] = None
) -> Versions
```

Returns the versions belonging to the registry.
