# Artifact

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.15.7/wandb/sdk/artifacts/artifact.py#L74-L2093' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


Flexible and lightweight building block for dataset and model versioning.

```python
Artifact(
    name: str,
    type: str,
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    incremental: bool = (False),
    use_as: Optional[str] = None
) -> None
```

Constructs an empty artifact whose contents can be populated using its `add` family
of functions. Once the artifact has all the desired files, you can call
`wandb.log_artifact()` to log it.

| Arguments |  |
| :--- | :--- |
|  `name` |  A human-readable name for this artifact, which is how you can identify this artifact in the UI or reference it in `use_artifact` calls. Names can contain letters, numbers, underscores, hyphens, and dots. The name must be unique across a project. |
|  `type` |  The type of the artifact, which is used to organize and differentiate artifacts. Common types include `dataset` or `model`, but you can use any string containing letters, numbers, underscores, hyphens, and dots. |
|  `description` |  Free text that offers a description of the artifact. The description is markdown rendered in the UI, so this is a good place to place tables, links, etc. |
|  `metadata` |  Structured data associated with the artifact, for example class distribution of a dataset. This will eventually be queryable and plottable in the UI. There is a hard limit of 100 total keys. |

| Returns |  |
| :--- | :--- |
|  An `Artifact` object. |

#### Examples:

Basic usage:

```
wandb.init()

artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_dir("mnist/")
wandb.log_artifact(artifact)
```

| Attributes |  |
| :--- | :--- |
|  `aliases` |  The aliases associated with this artifact. The list is mutable and calling `save()` will persist all alias changes. |
|  `commit_hash` |  The hash returned when this artifact was committed. |
|  `created_at` |  The time at which the artifact was created. |
|  `description` |  The artifact description. Free text that offers a user-set description of the artifact. |
|  `digest` |  The logical digest of the artifact. The digest is the checksum of the artifact's contents. If an artifact has the same digest as the current `latest` version, then `log_artifact` is a no-op. |
|  `entity` |  The name of the entity of the secondary (portfolio) artifact collection. |
|  `file_count` |  The number of files (including references). |
|  `id` |  The artifact's ID. |
|  `manifest` |  The artifact's manifest. The manifest lists all of its contents, and can't be changed once the artifact has been logged. |
|  `metadata` |  User-defined artifact metadata. Structured data associated with the artifact. |
|  `name` |  The artifact name and version in its secondary (portfolio) collection. A string with the format {collection}:{alias}. Before the artifact is saved, contains only the name since the version is not yet known. |
|  `project` |  The name of the project of the secondary (portfolio) artifact collection. |
|  `qualified_name` |  The entity/project/name of the secondary (portfolio) collection. |
|  `size` |  The total size of the artifact in bytes. Includes any references tracked by this artifact. |
|  `source_entity` |  The name of the entity of the primary (sequence) artifact collection. |
|  `source_name` |  The artifact name and version in its primary (sequence) collection. A string with the format {collection}:{alias}. Before the artifact is saved, contains only the name since the version is not yet known. |
|  `source_project` |  The name of the project of the primary (sequence) artifact collection. |
|  `source_qualified_name` |  The entity/project/name of the primary (sequence) collection. |
|  `source_version` |  The artifact's version in its primary (sequence) collection. A string with the format "v{number}". |
|  `state` |  The status of the artifact. One of: "PENDING", "COMMITTED", or "DELETED". |
|  `type` |  The artifact's type. |
|  `updated_at` |  The time at which the artifact was last updated. |
|  `version` |  The artifact's version in its secondary (portfolio) collection. |

## Methods

### `add`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.7/wandb/sdk/artifacts/artifact.py#L1222-L1337)

```python
add(
    obj: data_types.WBValue,
    name: StrPath
) -> ArtifactManifestEntry
```

Add wandb.WBValue `obj` to the artifact.

| Arguments |  |
| :--- | :--- |
|  `obj` |  The object to add. Currently support one of Bokeh, JoinedTable, PartitionedTable, Table, Classes, ImageMask, BoundingBoxes2D, Audio, Image, Video, Html, Object3D |
|  `name` |  The path within the artifact to add the object. |

| Returns |  |
| :--- | :--- |
|  The added manifest entry |

| Raises |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  if the artifact has already been finalized |

#### Examples:

Basic usage:

```
artifact = wandb.Artifact("my_table", type="dataset")
table = wandb.Table(
    columns=["a", "b", "c"],
    data=[(i, i * 2, 2**i) for i in range(10)]
)
artifact.add(table, "my_table")

wandb.log_artifact(artifact)
```

Retrieve an object:

```
artifact = wandb.use_artifact("my_table:latest")
table = artifact.get("my_table")
```

### `add_dir`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.7/wandb/sdk/artifacts/artifact.py#L1068-L1122)

```python
add_dir(
    local_path: str,
    name: Optional[str] = None
) -> None
```

Add a local directory to the artifact.

| Arguments |  |
| :--- | :--- |
|  `local_path` |  The path to the directory being added. |
|  `name` |  The path within the artifact to use for the directory being added. Defaults to the root of the artifact. |

| Raises |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  if the artifact has already been finalized |

#### Examples:

Add a directory without an explicit name:

```
# All files in `my_dir/` are added at the root of the artifact.
artifact.add_dir("my_dir/")
```

Add a directory and name it explicitly:

```
# All files in `my_dir/` are added under `destination/`.
artifact.add_dir("my_dir/", name="destination")
```

### `add_file`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.7/wandb/sdk/artifacts/artifact.py#L1019-L1066)

```python
add_file(
    local_path: str,
    name: Optional[str] = None,
    is_tmp: Optional[bool] = (False)
) -> ArtifactManifestEntry
```

Add a local file to the artifact.

| Arguments |  |
| :--- | :--- |
|  `local_path` |  The path to the file being added. |
|  `name` |  The path within the artifact to use for the file being added. Defaults to the basename of the file. |
|  `is_tmp` |  If true, then the file is renamed deterministically to avoid collisions. |

| Returns |  |
| :--- | :--- |
|  The added manifest entry |

| Raises |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  if the artifact has already been finalized |

#### Examples:

Add a file without an explicit name:

```
# Add as `file.txt'
artifact.add_file("path/to/file.txt")
```

Add a file with an explicit name:

```
# Add as 'new/path/file.txt'
artifact.add_file("path/to/file.txt", name="new/path/file.txt")
```

### `add_reference`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.7/wandb/sdk/artifacts/artifact.py#L1124-L1220)

```python
add_reference(
    uri: Union[ArtifactManifestEntry, str],
    name: Optional[StrPath] = None,
    checksum: bool = (True),
    max_objects: Optional[int] = None
) -> Sequence[ArtifactManifestEntry]
```

Add a reference denoted by a URI to the artifact.

Unlike adding files or directories, references are NOT uploaded to W&B. However,
artifact methods such as `download()` can be used regardless of whether the
artifact contains references or uploaded files.

By default, W&B offers special handling for the following schemes:

- http(s): The size and digest of the file will be inferred by the
  `Content-Length` and the `ETag` response headers returned by the server.
- s3: The checksum and size will be pulled from the object metadata. If bucket
  versioning is enabled, then the version ID is also tracked.
- gs: The checksum and size will be pulled from the object metadata. If bucket
  versioning is enabled, then the version ID is also tracked.
- https, domain matching *.blob.core.windows.net (Azure): The checksum and size
  will be pulled from the blob metadata. If storage account versioning is
  enabled, then the version ID is also tracked.
- file: The checksum and size will be pulled from the file system. This scheme
  is useful if you have an NFS share or other externally mounted volume
  containing files you wish to track but not necessarily upload.

For any other scheme, the digest is just a hash of the URI and the size is left
blank.

| Arguments |  |
| :--- | :--- |
|  `uri` |  The URI path of the reference to add. Can be an object returned from Artifact.get_path to store a reference to another artifact's entry. |
|  `name` |  The path within the artifact to place the contents of this reference |
|  `checksum` |  Whether or not to checksum the resource(s) located at the reference URI. Checksumming is strongly recommended as it enables automatic integrity validation, however it can be disabled to speed up artifact creation. (default: True) |
|  `max_objects` |  The maximum number of objects to consider when adding a reference that points to directory or bucket store prefix. For S3 and GCS, this limit is 10,000 by default but is uncapped for other URI schemes. (default: None) |

| Returns |  |
| :--- | :--- |
|  The added manifest entries. |

| Raises |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  if the artifact has already been finalized. |

#### Examples:

Add an HTTP link:

```python
# Adds `file.txt` to the root of the artifact as a reference.
artifact.add_reference("http://myserver.com/file.txt")
```

Add an S3 prefix without an explicit name:

```python
# All objects under `prefix/` will be added at the root of the artifact.
artifact.add_reference("s3://mybucket/prefix")
```

Add a GCS prefix with an explicit name:

```python
# All objects under `prefix/` will be added under `path/` at the artifact
# root.
artifact.add_reference("gs://mybucket/prefix", name="path")
```

### `checkout`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.7/wandb/sdk/artifacts/artifact.py#L1676-L1706)

```python
checkout(
    root: Optional[str] = None
) -> str
```

Replace the specified root directory with the contents of the artifact.

WARNING: This will DELETE all files in `root` that are not included in the
artifact.

| Arguments |  |
| :--- | :--- |
|  `root` |  The directory to replace with this artifact's files. |

| Returns |  |
| :--- | :--- |
|  The path to the checked out contents. |

| Raises |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  if the artifact has not been logged |

### `delete`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.7/wandb/sdk/artifacts/artifact.py#L1823-L1846)

```python
delete(
    delete_aliases: bool = (False)
) -> None
```

Delete an artifact and its files.

| Arguments |  |
| :--- | :--- |
|  `delete_aliases` |  If true, deletes all aliases associated with the artifact. Otherwise, this raises an exception if the artifact has existing aliases. |

| Raises |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  if the artifact has not been logged |

#### Examples:

Delete all the "model" artifacts a run has logged:

```
runs = api.runs(path="my_entity/my_project")
for run in runs:
    for artifact in run.logged_artifacts():
        if artifact.type == "model":
            artifact.delete(delete_aliases=True)
```

### `download`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.7/wandb/sdk/artifacts/artifact.py#L1535-L1641)

```python
download(
    root: Optional[str] = None,
    recursive: bool = (False),
    allow_missing_references: bool = (False)
) -> FilePathStr
```

Download the contents of the artifact to the specified root directory.

NOTE: Any existing files at `root` are left untouched. Explicitly delete
root before calling `download` if you want the contents of `root` to exactly
match the artifact.

| Arguments |  |
| :--- | :--- |
|  `root` |  The directory in which to download this artifact's files. |
|  `recursive` |  If true, then all dependent artifacts are eagerly downloaded. Otherwise, the dependent artifacts are downloaded as needed. |

| Returns |  |
| :--- | :--- |
|  The path to the downloaded contents. |

| Raises |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  if the artifact has not been logged |

### `file`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.7/wandb/sdk/artifacts/artifact.py#L1752-L1778)

```python
file(
    root: Optional[str] = None
) -> StrPath
```

Download a single file artifact to dir specified by the root.

| Arguments |  |
| :--- | :--- |
|  `root` |  The root directory in which to place the file. Defaults to './artifacts/self.name/'. |

| Returns |  |
| :--- | :--- |
|  The full path of the downloaded file. |

| Raises |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  if the artifact has not been logged |
|  `ValueError` |  if the artifact contains more than one file |

### `files`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.7/wandb/sdk/artifacts/artifact.py#L1780-L1799)

```python
files(
    names: Optional[List[str]] = None,
    per_page: int = 50
) -> ArtifactFiles
```

Iterate over all files stored in this artifact.

| Arguments |  |
| :--- | :--- |
|  `names` |  The filename paths relative to the root of the artifact you wish to list. |
|  `per_page` |  The number of files to return per request |

| Returns |  |
| :--- | :--- |
|  An iterator containing `File` objects |

| Raises |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  if the artifact has not been logged |

### `finalize`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.7/wandb/sdk/artifacts/artifact.py#L633-L638)

```python
finalize() -> None
```

Mark this artifact as final, disallowing further modifications.

This happens automatically when calling `log_artifact`.

### `get`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.7/wandb/sdk/artifacts/artifact.py#L1426-L1487)

```python
get(
    name: str
) -> Optional[data_types.WBValue]
```

Get the WBValue object located at the artifact relative `name`.

| Arguments |  |
| :--- | :--- |
|  `name` |  The artifact relative name to get |

| Raises |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  if the artifact isn't logged or the run is offline |

#### Examples:

Basic usage:

```
# Run logging the artifact
with wandb.init() as r:
    artifact = wandb.Artifact("my_dataset", type="dataset")
    table = wandb.Table(
        columns=["a", "b", "c"],
        data=[(i, i * 2, 2**i) for i in range(10)]
    )
    artifact.add(table, "my_table")
    wandb.log_artifact(artifact)

# Run using the artifact
with wandb.init() as r:
    artifact = r.use_artifact("my_dataset:latest")
    table = artifact.get("my_table")
```

### `get_added_local_path_name`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.7/wandb/sdk/artifacts/artifact.py#L1489-L1511)

```python
get_added_local_path_name(
    local_path: str
) -> Optional[str]
```

Get the artifact relative name of a file added by a local filesystem path.

| Arguments |  |
| :--- | :--- |
|  `local_path` |  The local path to resolve into an artifact relative name. |

| Returns |  |
| :--- | :--- |
|  The artifact relative name. |

#### Examples:

Basic usage:

```
artifact = wandb.Artifact("my_dataset", type="dataset")
artifact.add_file("path/to/file.txt", name="artifact/path/file.txt")

# Returns `artifact/path/file.txt`:
name = artifact.get_added_local_path_name("path/to/file.txt")
```

### `get_path`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.7/wandb/sdk/artifacts/artifact.py#L1388-L1424)

```python
get_path(
    name: StrPath
) -> ArtifactManifestEntry
```

Get the entry with the given name.

| Arguments |  |
| :--- | :--- |
|  `name` |  The artifact relative name to get |

| Raises |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  if the artifact isn't logged or the run is offline |
|  `KeyError` |  if the artifact doesn't contain an entry with the given name |

#### Examples:

Basic usage:

```
# Run logging the artifact
with wandb.init() as r:
    artifact = wandb.Artifact("my_dataset", type="dataset")
    artifact.add_file("path/to/file.txt")
    wandb.log_artifact(artifact)

# Run using the artifact
with wandb.init() as r:
    artifact = r.use_artifact("my_dataset:latest")
    path = artifact.get_path("file.txt")

    # Can now download 'file.txt' directly:
    path.download()
```

### `is_draft`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.7/wandb/sdk/artifacts/artifact.py#L644-L646)

```python
is_draft() -> bool
```

Whether the artifact is a draft, i.e. it hasn't been saved yet.

### `json_encode`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.7/wandb/sdk/artifacts/artifact.py#L2031-L2034)

```python
json_encode() -> Dict[str, Any]
```

### `link`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.7/wandb/sdk/artifacts/artifact.py#L1873-L1887)

```python
link(
    target_path: str,
    aliases: Optional[List[str]] = None
) -> None
```

Link this artifact to a portfolio (a promoted collection of artifacts).

| Arguments |  |
| :--- | :--- |
|  `target_path` |  The path to the portfolio. It must take the form {portfolio}, {project}/{portfolio} or {entity}/{project}/{portfolio}. |
|  `aliases` |  A list of strings which uniquely identifies the artifact inside the specified portfolio. |

| Raises |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  if the artifact has not been logged |

### `logged_by`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.7/wandb/sdk/artifacts/artifact.py#L1988-L2029)

```python
logged_by() -> Optional[Run]
```

Get the run that first logged this artifact.

| Raises |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  if the artifact has not been logged |

### `new_draft`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.7/wandb/sdk/artifacts/artifact.py#L339-L356)

```python
new_draft() -> "Artifact"
```

Create a new draft artifact with the same content as this committed artifact.

The artifact returned can be extended or modified and logged as a new version.

| Raises |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  if the artifact has not been logged |

### `new_file`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.7/wandb/sdk/artifacts/artifact.py#L973-L1017)

```python
@contextlib.contextmanager
new_file(
    name: str,
    mode: str = "w",
    encoding: Optional[str] = None
) -> Generator[IO, None, None]
```

Open a new temporary file that will be automatically added to the artifact.

| Arguments |  |
| :--- | :--- |
|  `name` |  The name of the new file being added to the artifact. |
|  `mode` |  The mode in which to open the new file. |
|  `encoding` |  The encoding in which to open the new file. |

| Returns |  |
| :--- | :--- |
|  A new file object that can be written to. Upon closing, the file will be automatically added to the artifact. |

| Raises |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  if the artifact has already been finalized. |

#### Examples:

```
artifact = wandb.Artifact("my_data", type="dataset")
with artifact.new_file("hello.txt") as f:
    f.write("hello!")
wandb.log_artifact(artifact)
```

### `remove`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.7/wandb/sdk/artifacts/artifact.py#L1358-L1386)

```python
remove(
    item: Union[StrPath, 'ArtifactManifestEntry']
) -> None
```

Remove an item from the artifact.

| Arguments |  |
| :--- | :--- |
|  `item` |  the item to remove. Can be a specific manifest entry or the name of an artifact-relative path. If the item matches a directory all items in that directory will be removed. |

| Raises |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  if the artifact has already been finalized. |
|  `FileNotFoundError` |  if the item isn't found in the artifact. |

### `save`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.7/wandb/sdk/artifacts/artifact.py#L651-L685)

```python
save(
    project: Optional[str] = None,
    settings: Optional['wandb.wandb_sdk.wandb_settings.Settings'] = None
) -> None
```

Persist any changes made to the artifact.

If currently in a run, that run will log this artifact. If not currently in a
run, a run of type "auto" will be created to track this artifact.

| Arguments |  |
| :--- | :--- |
|  `project` |  A project to use for the artifact in the case that a run is not already in context |
|  `settings` |  A settings object to use when initializing an automatic run. Most commonly used in testing harness. |

### `used_by`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.7/wandb/sdk/artifacts/artifact.py#L1943-L1986)

```python
used_by() -> List[Run]
```

Get a list of the runs that have used this artifact.

| Raises |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  if the artifact has not been logged |

### `verify`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.7/wandb/sdk/artifacts/artifact.py#L1708-L1750)

```python
verify(
    root: Optional[str] = None
) -> None
```

Verify that the actual contents of an artifact match the manifest.

All files in the directory are checksummed and the checksums are then
cross-referenced against the artifact's manifest.

NOTE: References are not verified.

| Arguments |  |
| :--- | :--- |
|  `root` |  The directory to verify. If None artifact will be downloaded to './artifacts/self.name/' |

| Raises |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  if the artifact has not been logged |
|  `ValueError` |  If the verification fails. |

### `wait`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.7/wandb/sdk/artifacts/artifact.py#L693-L711)

```python
wait(
    timeout: Optional[int] = None
) -> "Artifact"
```

Wait for this artifact to finish logging, if needed.

| Arguments |  |
| :--- | :--- |
|  `timeout` |  Wait up to this long. |

### `__getitem__`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.7/wandb/sdk/artifacts/artifact.py#L909-L937)

```python
__getitem__(
    name: str
) -> Optional[data_types.WBValue]
```

Get the WBValue object located at the artifact relative `name`.

| Arguments |  |
| :--- | :--- |
|  `name` |  The artifact relative name to get |

| Raises |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  if the artifact isn't logged or the run is offline |

#### Examples:

Basic usage:

```
artifact = wandb.Artifact("my_table", type="dataset")
table = wandb.Table(
    columns=["a", "b", "c"],
    data=[(i, i * 2, 2**i) for i in range(10)]
)
artifact["my_table"] = table

wandb.log_artifact(artifact)
```

Retrieving an object:

```
artifact = wandb.use_artifact("my_table:latest")
table = artifact["my_table"]
```

### `__setitem__`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.7/wandb/sdk/artifacts/artifact.py#L939-L971)

```python
__setitem__(
    name: str,
    item: data_types.WBValue
) -> ArtifactManifestEntry
```

Add `item` to the artifact at path `name`.

| Arguments |  |
| :--- | :--- |
|  `name` |  The path within the artifact to add the object. |
|  `item` |  The object to add. |

| Returns |  |
| :--- | :--- |
|  The added manifest entry |

| Raises |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  if the artifact has already been finalized. |

#### Examples:

Basic usage:

```
artifact = wandb.Artifact("my_table", type="dataset")
table = wandb.Table(
    columns=["a", "b", "c"],
    data=[(i, i * 2, 2**i) for i in range(10)]
)
artifact["my_table"] = table

wandb.log_artifact(artifact)
```

Retrieving an object:

```
artifact = wandb.use_artifact("my_table:latest")
table = artifact["my_table"]
```
