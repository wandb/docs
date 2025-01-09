---
title: Artifact
---

{{< cta-button githubLink="https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/sdk/artifacts/artifact.py#L84-L2301">}}


Flexible and lightweight building block for dataset and model versioning.

```python
Artifact(
    name: str,
    type: str,
    description: (str | None) = None,
    metadata: (dict[str, Any] | None) = None,
    incremental: bool = (False),
    use_as: (str | None) = None
) -> None
```

Construct an empty W&B Artifact. Populate an artifacts contents with methods that
begin with `add`. Once the artifact has all the desired files, you can call
`wandb.log_artifact()` to log it.

| Args |  |
| :--- | :--- |
| `name` | A human-readable name for the artifact. Use the name to identify a specific artifact in the W&B App UI or programmatically. You can interactively reference an artifact with the `use_artifact` Public API. A name can contain letters, numbers, underscores, hyphens, and dots. The name must be unique across a project. |
| `type` | The artifact's type. Use the type of an artifact to both organize and differentiate artifacts. You can use any string that contains letters, numbers, underscores, hyphens, and dots. Common types include `dataset` or `model`. Include `model` within your type string if you want to link the artifact to the W&B Model Registry. |
| `description` | A description of the artifact. For Model or Dataset Artifacts, add documentation for your standardized team model or dataset card. View an artifact's description programmatically with the `Artifact.description` attribute or programmatically with the W&B App UI. W&B renders the description as markdown in the W&B App. |
| `metadata` | Additional information about an artifact. Specify metadata as a dictionary of key-value pairs. You can specify no more than 100 total keys. |

| Returns |  |
| :--- | :--- |
| An `Artifact` object. |

| Attributes |  |
| :--- | :--- |
| `aliases` |  List of one or more semantically friendly references or identifying "nicknames" assigned to an artifact version. Aliases are mutable references that you can programmatically reference. Change an artifact's alias with the W&B App UI or programmatically. See [Create new artifact versions](https://docs.wandb.ai/guides/artifacts/create-a-new-artifact-version) for more information. |
| `collection` |  The collection this artifact was retrieved from. A collection is an ordered group of artifact versions. If this artifact was retrieved from a portfolio / linked collection, that collection will be returned rather than the collection that an artifact version originated from. The collection that an artifact originates from is known as the source sequence. |
| `commit_hash` |  The hash returned when this artifact was committed. |
| `created_at` |  Timestamp when the artifact was created. |
| `description` |  A description of the artifact. |
| `digest` |  The logical digest of the artifact. The digest is the checksum of the artifact's contents. If an artifact has the same digest as the current `latest` version, then `log_artifact` is a no-op. |
| `entity` |  The name of the entity of the secondary (portfolio) artifact collection. |
| `file_count` |  The number of files (including references). |
| `id` |  The artifact's ID. |
| `manifest` |  The artifact's manifest. The manifest lists all of its contents, and can't be changed once the artifact has been logged. |
| `metadata` |  User-defined artifact metadata. Structured data associated with the artifact. |
| `name` |  The artifact name and version in its secondary (portfolio) collection. A string with the format `{collection}:{alias}`. Before the artifact is saved, contains only the name since the version is not yet known. |
| `project` |  The name of the project of the secondary (portfolio) artifact collection. |
| `qualified_name` |  The entity/project/name of the secondary (portfolio) collection. |
| `size` |  The total size of the artifact in bytes. Includes any references tracked by this artifact. |
| `source_collection` |  The artifact's primary (sequence) collection. |
| `source_entity` |  The name of the entity of the primary (sequence) artifact collection. |
| `source_name` |  The artifact name and version in its primary (sequence) collection. A string with the format `{collection}:{alias}`. Before the artifact is saved, contains only the name since the version is not yet known. |
| `source_project` |  The name of the project of the primary (sequence) artifact collection. |
| `source_qualified_name` |  The entity/project/name of the primary (sequence) collection. |
| `source_version` |  The artifact's version in its primary (sequence) collection. A string with the format `v{number}`. |
| `state` |  The status of the artifact. One of: `PENDING`, `COMMITTED`, or `DELETED`. |
| `tags` |  List of one or more tags assigned to this artifact version. |
| `ttl` |  The time-to-live (TTL) policy of an artifact. Artifacts are deleted shortly after a TTL policy's duration passes. If set to `None`, the artifact deactivates TTL policies and will be not scheduled for deletion, even if there is a team default TTL. An artifact inherits a TTL policy from the team default if the team administrator defines a default TTL and there is no custom policy set on an artifact. |
| `type` |  The artifact's type. Common types include `dataset` or `model`. |
| `updated_at` |  The time when the artifact was last updated. |
| `version` |  The artifact's version in its secondary (portfolio) collection. |

## Methods

### `add`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/sdk/artifacts/artifact.py#L1354-L1445)

```python
add(
    obj: WBValue,
    name: StrPath,
    overwrite: bool = (False)
) -> ArtifactManifestEntry
```

Add `wandb.WBValue` `obj` to the artifact.

| Args |  |
| :--- | :--- |
| `obj` | The object to add. Currently support one of `Bokeh`, `JoinedTable`, `PartitionedTable`, `Table`, `Classes`, `ImageMask`, `BoundingBoxes2D`, `Audio`, `Image`, `Video`, `Html`, `Object3D` |
| `name` | The path within the artifact to add the object. |
| `overwrite` |  If True, overwrite existing objects with the same file path (if applicable). |

| Returns |  |
| :--- | :--- |
| The added manifest entry |

| Raises |  |
| :--- | :--- |
| `ArtifactFinalizedError` | You cannot make changes to the current artifact version because it is finalized. Log a new artifact version instead. |

### `add_dir`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/sdk/artifacts/artifact.py#L1209-L1269)

```python
add_dir(
    local_path: str,
    name: (str | None) = None,
    skip_cache: (bool | None) = (False),
    policy: (Literal['mutable', 'immutable'] | None) = "mutable"
) -> None
```

Add a local directory to the artifact.

| Args |  |
| :--- | :--- |
| `local_path` |  The path of the local directory. |
| `name` |  The subdirectory name within an artifact. The name you specify appears in the W&B App UI nested by artifact's `type`. Defaults to the root of the artifact. |
| `skip_cache` |  If set to `True`, W&B will not copy/move files to the cache while uploading |
| `policy` |  "mutable" | "immutable". By default, "mutable" "mutable": Create a temporary copy of the file to prevent corruption during upload. "immutable": Disable protection, rely on the user not to delete or change the file. |

| Raises |  |
| :--- | :--- |
| `ArtifactFinalizedError` |  You cannot make changes to the current artifact version because it is finalized. Log a new artifact version instead. |
| `ValueError` |  Policy must be "mutable" or "immutable" |

### `add_file`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/sdk/artifacts/artifact.py#L1156-L1207)

```python
add_file(
    local_path: str,
    name: (str | None) = None,
    is_tmp: (bool | None) = (False),
    skip_cache: (bool | None) = (False),
    policy: (Literal['mutable', 'immutable'] | None) = "mutable",
    overwrite: bool = (False)
) -> ArtifactManifestEntry
```

Add a local file to the artifact.

| Args |  |
| :--- | :--- |
| `local_path` |  The path to the file being added. |
| `name` |  The path within the artifact to use for the file being added. Defaults to the basename of the file. |
| `is_tmp` |  If true, then the file is renamed deterministically to avoid collisions. |
| `skip_cache` |  If `True`, W&B will not copy files to the cache after uploading. |
| `policy` | By default, set to `mutable`. If set to `mutable`, create a temporary copy of the file to prevent corruption during upload. If set to `immutable`, disable protection and rely on the user not to delete or change the file. |
| `overwrite` |  If `True`, overwrite the file if it already exists. |

| Returns |  |
| :--- | :--- |
| The added manifest entry. |

| Raises |  |
| :--- | :--- |
| `ArtifactFinalizedError` |  You cannot make changes to the current artifact version because it is finalized. Log a new artifact version instead. |
| `ValueError` |  Policy must be "mutable" or "immutable" |

### `add_reference`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/sdk/artifacts/artifact.py#L1271-L1352)

```python
add_reference(
    uri: (ArtifactManifestEntry | str),
    name: (StrPath | None) = None,
    checksum: bool = (True),
    max_objects: (int | None) = None
) -> Sequence[ArtifactManifestEntry]
```

Add a reference denoted by a URI to the artifact.

Unlike files or directories that you add to an artifact, references are not
uploaded to W&B. For more information,
see [Track external files](https://docs.wandb.ai/guides/artifacts/track-external-files).

By default, the following schemes are supported:

- `http` or `https`: The size and digest of the file will be inferred by the `Content-Length` and the `ETag` response headers returned by the server.
-  `s3`: The checksum and size are pulled from the object metadata. If bucket versioning is enabled, then the version ID is also tracked.
- `gs`: The checksum and size are pulled from the object metadata. If bucket versioning is enabled, then the version ID is also tracked.
- `https`, domain matching `*.blob.core.windows.net` (Azure): The checksum and size are be pulled from the blob metadata. If storage account versioning is enabled, then the version ID is also tracked.
- `file`: The checksum and size are pulled from the file system. This scheme is useful if you have an NFS share or other externally mounted volume containing files you wish to track but not necessarily upload

For any other scheme, the digest is just a hash of the URI and the size is left
blank.

| Args |  |
| :--- | :--- |
| `uri` |  The URI path of the reference to add. The URI path can be an object returned from `Artifact.get_entry` to store a reference to another artifact's entry. |
| `name` |  The path within the artifact to place the contents of this reference. |
| `checksum` |  Whether or not to checksum the resources located at the reference URI. Checksumming is strongly recommended as it enables automatic integrity validation. Disabling checksumming will speed up artifact creation but reference directories will not iterated through so the objects in the directory will not be saved to the artifact. We recommend setting `checksum=False` when adding reference objects, in which case a new version will only be created if the reference URI changes. |
| `max_objects` |  The maximum number of objects to consider when adding a reference that points to directory or bucket store prefix. By default, the maximum number of objects allowed for Amazon S3, GCS, Azure, and local files is 10,000,000. Other URI schemas do not have a maximum. |

| Returns |  |
| :--- | :--- |
| The added manifest entries. |

| Raises |  |
| :--- | :--- |
| `ArtifactFinalizedError` | You cannot make changes to the current artifact version because it is finalized. Log a new artifact version instead. |

### `checkout`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/sdk/artifacts/artifact.py#L1870-L1898)

```python
checkout(
    root: (str | None) = None
) -> str
```

Replace the specified root directory with the contents of the artifact.

{{% alert title="Warning" color="warning" %}}
This will delete all files in `root` that are not included in the artifact.
{{% /alert %}}


| Args |  |
| :--- | :--- |
| `root` | The directory to replace with this artifact's files. |

| Returns |  |
| :--- | :--- |
| The path of the checked out contents. |

| Raises |  |
| :--- | :--- |
| `ArtifactNotLoggedError` | If the artifact is not logged. |

### `delete`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/sdk/artifacts/artifact.py#L2008-L2027)

```python
delete(
    delete_aliases: bool = (False)
) -> None
```

Delete an artifact and its files.

If called on a linked artifact, such as a member of a portfolio collection, deletes only the link, not the source artifact.

| Args |  |
| :--- | :--- |
| `delete_aliases` | If set to `True`, deletes all aliases associated with the artifact. Otherwise, raises an exception if the artifact has existing aliases. Ignored if the artifact is linked, such as if it is a member of a portfolio collection. |

| Raises |  |
| :--- | :--- |
| `ArtifactNotLoggedError` | If the artifact is not logged. |

### `download`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/sdk/artifacts/artifact.py#L1623-L1674)

```python
download(
    root: (StrPath | None) = None,
    allow_missing_references: bool = (False),
    skip_cache: (bool | None) = None,
    path_prefix: (StrPath | None) = None
) -> FilePathStr
```

Download the contents of the artifact to the specified root directory.

Existing files located within `root` are not modified. Explicitly delete `root`
before you call `download` if you want the contents of `root` to exactly match
the artifact.

| Args |  |
| :--- | :--- |
| `root` | The directory W&B stores the artifact's files. |
| `allow_missing_references` | If set to `True`, any invalid reference paths will be ignored while downloading referenced files. |
| `skip_cache` | If set to `True`, the artifact cache will be skipped when downloading and W&B will download each file into the default root or specified download directory. |
| `path_prefix` | If specified, only files with a path that starts with the given prefix will be downloaded. Uses unix format (forward slashes). |

| Returns |  |
| :--- | :--- |
| The path to the downloaded contents. |

| Raises |  |
| :--- | :--- |
| `ArtifactNotLoggedError` | If the artifact is not logged. |
| `RuntimeError` | If the artifact is attempted to be downloaded in offline mode. |

### `file`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/sdk/artifacts/artifact.py#L1940-L1964)

```python
file(
    root: (str | None) = None
) -> StrPath
```

Download a single file artifact to the directory you specify with `root`.

| Args |  |
| :--- | :--- |
| `root` | The root directory to store the file. Defaults to './artifacts/self.name/'. |

| Returns |  |
| :--- | :--- |
| The full path of the downloaded file. |

| Raises |  |
| :--- | :--- |
| `ArtifactNotLoggedError` | If the artifact is not logged. |
| `ValueError` | If the artifact contains more than one file. |

### `files`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/sdk/artifacts/artifact.py#L1966-L1983)

```python
files(
    names: (list[str] | None) = None,
    per_page: int = 50
) -> ArtifactFiles
```

Iterate over all files stored in this artifact.

| Args |  |
| :--- | :--- |
| `names` | The filename paths relative to the root of the artifact you wish to list. |
| `per_page` | The number of files to return per request. |

| Returns |  |
| :--- | :--- |
| An iterator containing `File` objects. |

| Raises |  |
| :--- | :--- |
| `ArtifactNotLoggedError` | If the artifact is not logged. |

### `finalize`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/sdk/artifacts/artifact.py#L774-L782)

```python
finalize() -> None
```

Finalize the artifact version.

You cannot modify an artifact version once it is finalized because the artifact
is logged as a specific artifact version. Create a new artifact version
to log more data to an artifact. An artifact is automatically finalized
when you log the artifact with `log_artifact`.

### `get`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/sdk/artifacts/artifact.py#L1540-L1585)

```python
get(
    name: str
) -> (WBValue | None)
```

Get the WBValue object located at the artifact relative `name`.

| Args |  |
| :--- | :--- |
| `name` | The artifact relative name to retrieve. |

| Returns |  |
| :--- | :--- |
| W&B object that can be logged with `wandb.log()` and visualized in the W&B UI. |

| Raises |  |
| :--- | :--- |
| `ArtifactNotLoggedError` | if the artifact isn't logged or the run is offline |

### `get_added_local_path_name`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/sdk/artifacts/artifact.py#L1587-L1599)

```python
get_added_local_path_name(
    local_path: str
) -> (str | None)
```

Get the artifact relative name of a file added by a local filesystem path.

| Args |  |
| :--- | :--- |
| `local_path` | The local path to resolve into an artifact relative name. |

| Returns |  |
| :--- | :--- |
| The artifact relative name. |

### `get_entry`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/sdk/artifacts/artifact.py#L1519-L1538)

```python
get_entry(
    name: StrPath
) -> ArtifactManifestEntry
```

Get the entry with the given name.

| Args |  |
| :--- | :--- |
| `name` | The artifact relative name to get |

| Returns |  |
| :--- | :--- |
| A `W&B` object. |

| Raises |  |
| :--- | :--- |
| `ArtifactNotLoggedError` | if the artifact isn't logged or the run is offline. |
| `KeyError` | if the artifact doesn't contain an entry with the given name. |

### `get_path`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/sdk/artifacts/artifact.py#L1511-L1517)

```python
get_path(
    name: StrPath
) -> ArtifactManifestEntry
```

Deprecated. Use `get_entry(name)`.

### `is_draft`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/sdk/artifacts/artifact.py#L784-L789)

```python
is_draft() -> bool
```

Check if artifact is not saved.

Returns: Boolean. `False` if artifact is saved. `True` if artifact is not saved.

### `json_encode`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/sdk/artifacts/artifact.py#L2215-L2222)

```python
json_encode() -> dict[str, Any]
```

Returns the artifact encoded to the JSON format.

| Returns |  |
| :--- | :--- |
| A `dict` with `string` keys representing attributes of the artifact. |

### `link`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/sdk/artifacts/artifact.py#L2054-L2082)

```python
link(
    target_path: str,
    aliases: (list[str] | None) = None
) -> None
```

Link this artifact to a portfolio (a promoted collection of artifacts).

| Args |  |
| :--- | :--- |
| `target_path` | The path to the portfolio inside a project. The target path must adhere to one of the following schemas `{portfolio}`, `{project}/{portfolio}` or `{entity}/{project}/{portfolio}`. To link the artifact to the Model Registry, rather than to a generic portfolio inside a project, set `target_path` to the following schema `{"model-registry"}/{Registered Model Name}` or `{entity}/{"model-registry"}/{Registered Model Name}`. |
| `aliases` | A list of strings that uniquely identifies the artifact inside the specified portfolio. |

| Raises |  |
| :--- | :--- |
| `ArtifactNotLoggedError` | If the artifact is not logged. |

### `logged_by`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/sdk/artifacts/artifact.py#L2171-L2213)

```python
logged_by() -> (Run | None)
```

Get the W&B run that originally logged the artifact.

| Returns |  |
| :--- | :--- |
| The name of the W&B run that originally logged the artifact. |

| Raises |  |
| :--- | :--- |
| `ArtifactNotLoggedError` | If the artifact is not logged. |

### `new_draft`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/sdk/artifacts/artifact.py#L389-L420)

```python
new_draft() -> Artifact
```

Create a new draft artifact with the same content as this committed artifact.

The artifact returned can be extended or modified and logged as a new version.

| Returns |  |
| :--- | :--- |
| An `Artifact` object. |

| Raises |  |
| :--- | :--- |
| `ArtifactNotLoggedError` | If the artifact is not logged. |

### `new_file`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/sdk/artifacts/artifact.py#L1113-L1154)

```python
@contextlib.contextmanager
new_file(
    name: str,
    mode: str = "x",
    encoding: (str | None) = None
) -> Iterator[IO]
```

Open a new temporary file and add it to the artifact.

| Args |  |
| :--- | :--- |
| `name` | The name of the new file to add to the artifact. |
| `mode` | The file access mode to use to open the new file. |
| `encoding` | The encoding used to open the new file. |

| Returns |  |
| :--- | :--- |
| A new file object that can be written to. Upon closing, the file will be automatically added to the artifact. |

| Raises |  |
| :--- | :--- |
| `ArtifactFinalizedError` | You cannot make changes to the current artifact version because it is finalized. Log a new artifact version instead. |

### `remove`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/sdk/artifacts/artifact.py#L1481-L1509)

```python
remove(
    item: (StrPath | ArtifactManifestEntry)
) -> None
```

Remove an item from the artifact.

| Args |  |
| :--- | :--- |
| `item` | The item to remove. Can be a specific manifest entry or the name of an artifact-relative path. If the item matches a directory all items in that directory will be removed. |

| Raises |  |
| :--- | :--- |
| `ArtifactFinalizedError` | You cannot make changes to the current artifact version because it is finalized. Log a new artifact version instead. |
| `FileNotFoundError` | If the item isn't found in the artifact. |

### `save`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/sdk/artifacts/artifact.py#L794-L833)

```python
save(
    project: (str | None) = None,
    settings: (wandb.Settings | None) = None
) -> None
```

Persist any changes made to the artifact.

If currently in a run, that run will log this artifact. If not currently in a
run, a run of type "auto" is created to track this artifact.

| Args |  |
| :--- | :--- |
| `project` | A project to use for the artifact in the case that a run is not already in context. |
| `settings` | A settings object to use when initializing an automatic run. Most commonly used in testing harness. |

### `unlink`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/sdk/artifacts/artifact.py#L2084-L2099)

```python
unlink() -> None
```

Unlink this artifact if it is currently a member of a portfolio (a promoted collection of artifacts).

| Raises |  |
| :--- | :--- |
| `ArtifactNotLoggedError` | If the artifact is not logged. |
| `ValueError` | If the artifact is not linked, such as if it is not a member of a portfolio collection. |

### `used_by`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/sdk/artifacts/artifact.py#L2125-L2169)

```python
used_by() -> list[Run]
```

Get a list of the runs that have used this artifact.

| Returns |  |
| :--- | :--- |
| A list of `Run` objects. |

| Raises |  |
| :--- | :--- |
| `ArtifactNotLoggedError` | If the artifact is not logged. |

### `verify`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/sdk/artifacts/artifact.py#L1900-L1938)

```python
verify(
    root: (str | None) = None
) -> None
```

Verify that the contents of an artifact match the manifest.

All files in the directory are checksummed and the checksums are then
cross-referenced against the artifact's manifest. References are not verified.

| Args |  |
| :--- | :--- |
| `root` | The directory to verify. If None artifact will be downloaded to './artifacts/self.name/' |

| Raises |  |
| :--- | :--- |
| `ArtifactNotLoggedError` | If the artifact is not logged. |
| `ValueError` | If the verification fails. |

### `wait`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/sdk/artifacts/artifact.py#L841-L862)

```python
wait(
    timeout: (int | None) = None
) -> Artifact
```

If needed, wait for this artifact to finish logging.

| Args |  |
| :--- | :--- |
| `timeout` | The time, in seconds, to wait. |

| Returns |  |
| :--- | :--- |
| An `Artifact` object. |

### `__getitem__`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/sdk/artifacts/artifact.py#L1083-L1095)

```python
__getitem__(
    name: str
) -> (WBValue | None)
```

Get the WBValue object located at the artifact relative `name`.

| Args |  |
| :--- | :--- |
| `name` | The artifact relative name to get. |

| Returns |  |
| :--- | :--- |
| W&B object that can be logged with `wandb.log()` and visualized in the W&B UI. |

| Raises |  |
| :--- | :--- |
| `ArtifactNotLoggedError` | If the artifact isn't logged or the run is offline. |

### `__setitem__`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/sdk/artifacts/artifact.py#L1097-L1111)

```python
__setitem__(
    name: str,
    item: WBValue
) -> ArtifactManifestEntry
```

Add `item` to the artifact at path `name`.

| Args |  |
| :--- | :--- |
| `name` | The path within the artifact to add the object. |
| `item` | The object to add. |

| Returns |  |
| :--- | :--- |
| The added manifest entry |

| Raises |  |
| :--- | :--- |
| `ArtifactFinalizedError` | You cannot make changes to the current artifact version because it is finalized. Log a new artifact version instead. |
