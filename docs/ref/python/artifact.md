# Artifact



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/local_artifact.py#L70-L719)



Flexible and lightweight building block for dataset and model versioning.

```python
Artifact(
 name: str,
 type: str,
 description: Optional[str] = None,
 metadata: Optional[dict] = None,
 incremental: Optional[bool] = None,
 use_as: Optional[str] = None
) -> None
```




Constructs an empty artifact whose contents can be populated using its
`add` family of functions. Once the artifact has all the desired files,
you can call `wandb.log_artifact()` to log it.

| Arguments | |
| :--- | :--- |
| `name` | (str) A human-readable name for this artifact, which is how you can identify this artifact in the UI or reference it in `use_artifact` calls. Names can contain letters, numbers, underscores, hyphens, and dots. The name must be unique across a project. |
| `type` | (str) The type of the artifact, which is used to organize and differentiate artifacts. Common types include `dataset` or `model`, but you can use any string containing letters, numbers, underscores, hyphens, and dots. |
| `description` | (str, optional) Free text that offers a description of the artifact. The description is markdown rendered in the UI, so this is a good place to place tables, links, etc. |
| `metadata` | (dict, optional) Structured data associated with the artifact, for example class distribution of a dataset. This will eventually be queryable and plottable in the UI. There is a hard limit of 100 total keys. |



#### Examples:

Basic usage
```
wandb.init()

artifact = wandb.Artifact('mnist', type='dataset')
artifact.add_dir('mnist/')
wandb.log_artifact(artifact)
```



| Returns | |
| :--- | :--- |
| An `Artifact` object. |





| Attributes | |
| :--- | :--- |
| `aliases` | The aliases associated with this artifact. The list is mutable and calling `save()` will persist all alias changes. |
| `commit_hash` | The hash returned when this artifact was committed. |
| `description` | The artifact description. |
| `digest` | The logical digest of the artifact. The digest is the checksum of the artifact's contents. If an artifact has the same digest as the current `latest` version, then `log_artifact` is a no-op. |
| `entity` | The name of the entity of the secondary (portfolio) artifact collection. |
| `id` | The artifact's ID. |
| `manifest` | The artifact's manifest. The manifest lists all of its contents, and can't be changed once the artifact has been logged. |
| `metadata` | User-defined artifact metadata. |
| `name` | The artifact name and version in its secondary (portfolio) collection. A string with the format {collection}:{alias}. Before the artifact is saved, contains only the name since the version is not yet known. |
| `project` | The name of the project of the secondary (portfolio) artifact collection. |
| `qualified_name` | The entity/project/name of the secondary (portfolio) collection. |
| `size` | The total size of the artifact in bytes. |
| `source_entity` | The name of the entity of the primary (sequence) artifact collection. |
| `source_name` | The artifact name and version in its primary (sequence) collection. A string with the format {collection}:{alias}. Before the artifact is saved, contains only the name since the version is not yet known. |
| `source_project` | The name of the project of the primary (sequence) artifact collection. |
| `source_qualified_name` | The entity/project/name of the primary (sequence) collection. |
| `source_version` | The artifact's version in its primary (sequence) collection. A string with the format "v{number}". |
| `state` | The status of the artifact. One of: "PENDING", "COMMITTED", or "DELETED". |
| `type` | The artifact's type. |
| `version` | The artifact's version in its secondary (portfolio) collection. A string with the format "v{number}". |



## Methods

### `add`



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/local_artifact.py#L462-L543)

```python
add(
 obj: WBValue,
 name: StrPath
) -> ArtifactManifestEntry
```

Add wandb.WBValue `obj` to the artifact.

```
obj = artifact.get(name)
```

| Arguments | |
| :--- | :--- |
| `obj` | (wandb.WBValue) The object to add. Currently support one of Bokeh, JoinedTable, PartitionedTable, Table, Classes, ImageMask, BoundingBoxes2D, Audio, Image, Video, Html, Object3D |
| `name` | (str) The path within the artifact to add the object. |



| Returns | |
| :--- | :--- |
| `ArtifactManifestEntry` | the added manifest entry |



| Raises | |
| :--- | :--- |
| `ArtifactFinalizedError` | if the artifact has already been finalized. |



#### Examples:

Basic usage
```
artifact = wandb.Artifact('my_table', 'dataset')
table = wandb.Table(columns=["a", "b", "c"], data=[[i, i*2, 2**i]])
artifact.add(table, "my_table")

wandb.log_artifact(artifact)
```

Retrieve an object:
```
artifact = wandb.use_artifact('my_table:latest')
table = artifact.get("my_table")
```


### `add_dir`



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/local_artifact.py#L390-L423)

```python
add_dir(
 local_path: str,
 name: Optional[str] = None
) -> None
```

Add a local directory to the artifact.


| Arguments | |
| :--- | :--- |
| `local_path` | (str) The path to the directory being added. |
| `name` | (str, optional) The path within the artifact to use for the directory being added. Defaults to the root of the artifact. |



#### Examples:

Add a directory without an explicit name:
```
# All files in `my_dir/` are added at the root of the artifact.
artifact.add_dir('my_dir/')
```

Add a directory and name it explicitly:
```
# All files in `my_dir/` are added under `destination/`.
artifact.add_dir('my_dir/', name='destination')
```



| Raises | |
| :--- | :--- |
| `ArtifactFinalizedError` | if the artifact has already been finalized. |



| Returns | |
| :--- | :--- |
| None |



### `add_file`



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/local_artifact.py#L369-L388)

```python
add_file(
 local_path: str,
 name: Optional[str] = None,
 is_tmp: Optional[bool] = False
) -> ArtifactManifestEntry
```

Add a local file to the artifact.


| Arguments | |
| :--- | :--- |
| `local_path` | (str) The path to the file being added. |
| `name` | (str, optional) The path within the artifact to use for the file being added. Defaults to the basename of the file. |
| `is_tmp` | (bool, optional) If true, then the file is renamed deterministically to avoid collisions. (default: False) |



#### Examples:

Add a file without an explicit name:
```
# Add as `file.txt'
artifact.add_file('path/to/file.txt')
```

Add a file with an explicit name:
```
# Add as 'new/path/file.txt'
artifact.add_file('path/to/file.txt', name='new/path/file.txt')
```



| Raises | |
| :--- | :--- |
| `ArtifactFinalizedError` | if the artifact has already been finalized. |



| Returns | |
| :--- | :--- |
| `ArtifactManifestEntry` | the added manifest entry |



### `add_reference`



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/local_artifact.py#L425-L460)

```python
add_reference(
 uri: Union[ArtifactManifestEntry, str],
 name: Optional[StrPath] = None,
 checksum: bool = True,
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

| Arguments | |
| :--- | :--- |
| `uri` | (str) The URI path of the reference to add. Can be an object returned from Artifact.get_path to store a reference to another artifact's entry. |
| `name` | (str) The path within the artifact to place the contents of this reference |
| `checksum` | (bool, optional) Whether or not to checksum the resource(s) located at the reference URI. Checksumming is strongly recommended as it enables automatic integrity validation, however it can be disabled to speed up artifact creation. (default: True) |
| `max_objects` | (int, optional) The maximum number of objects to consider when adding a reference that points to directory or bucket store prefix. For S3 and GCS, this limit is 10,000 by default but is uncapped for other URI schemes. (default: None) |



| Raises | |
| :--- | :--- |
| `ArtifactFinalizedError` | if the artifact has already been finalized. |



| Returns | |
| :--- | :--- |
| List["ArtifactManifestEntry"]: The added manifest entries. |



#### Examples:



#### Add an HTTP link:


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
# All objects under `prefix/` will be added under `path/` at the artifact root.
artifact.add_reference("gs://mybucket/prefix", name="path")
```

### `checkout`



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/local_artifact.py#L585-L589)

```python
checkout(
 root: Optional[str] = None
) -> str
```

Replace the specified root directory with the contents of the artifact.

WARNING: This will DELETE all files in `root` that are not included in the
artifact.

| Arguments | |
| :--- | :--- |
| `root` | (str, optional) The directory to replace with this artifact's files. |



| Returns | |
| :--- | :--- |
| (str): The path to the checked out contents. |



### `delete`



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/local_artifact.py#L638-L642)

```python
delete() -> None
```

Delete this artifact, cleaning up all files associated with it.

NOTE: Deletion is permanent and CANNOT be undone.

| Returns | |
| :--- | :--- |
| None |



### `download`



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/local_artifact.py#L577-L583)

```python
download(
 root: Optional[str] = None,
 recursive: bool = False
) -> FilePathStr
```

Download the contents of the artifact to the specified root directory.

NOTE: Any existing files at `root` are left untouched. Explicitly delete
root before calling `download` if you want the contents of `root` to exactly
match the artifact.

| Arguments | |
| :--- | :--- |
| `root` | (str, optional) The directory in which to download this artifact's files. |
| `recursive` | (bool, optional) If true, then all dependent artifacts are eagerly downloaded. Otherwise, the dependent artifacts are downloaded as needed. |



| Returns | |
| :--- | :--- |
| (str): The path to the downloaded contents. |



### `finalize`



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/local_artifact.py#L679-L692)

```python
finalize() -> None
```

Mark this artifact as final, disallowing further modifications.

This happens automatically when calling `log_artifact`.

| Returns | |
| :--- | :--- |
| None |



### `get`



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/local_artifact.py#L571-L575)

```python
get(
 name: str
) -> Optional[WBValue]
```

Get the WBValue object located at the artifact relative `name`.


| Arguments | |
| :--- | :--- |
| `name` | (str) The artifact relative name to get |



| Raises | |
| :--- | :--- |
| `ArtifactNotLoggedError` | if the artifact isn't logged or the run is offline |



#### Examples:

Basic usage
```
# Run logging the artifact
with wandb.init() as r:
 artifact = wandb.Artifact('my_dataset', type='dataset')
 table = wandb.Table(columns=["a", "b", "c"], data=[[i, i*2, 2**i]])
 artifact.add(table, "my_table")
 wandb.log_artifact(artifact)

# Run using the artifact
with wandb.init() as r:
 artifact = r.use_artifact('my_dataset:latest')
 table = r.get('my_table')
```


### `get_added_local_path_name`



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/local_artifact.py#L655-L677)

```python
get_added_local_path_name(
 local_path: str
) -> Optional[str]
```

Get the artifact relative name of a file added by a local filesystem path.


| Arguments | |
| :--- | :--- |
| `local_path` | (str) The local path to resolve into an artifact relative name. |



| Returns | |
| :--- | :--- |
| `str` | The artifact relative name. |



#### Examples:

Basic usage
```
artifact = wandb.Artifact('my_dataset', type='dataset')
artifact.add_file('path/to/file.txt', name='artifact/path/file.txt')

# Returns `artifact/path/file.txt`:
name = artifact.get_added_local_path_name('path/to/file.txt')
```


### `get_path`



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/local_artifact.py#L565-L569)

```python
get_path(
 name: StrPath
) -> ArtifactManifestEntry
```

Get the path to the file located at the artifact relative `name`.


| Arguments | |
| :--- | :--- |
| `name` | (str) The artifact relative name to get |



| Raises | |
| :--- | :--- |
| `ArtifactNotLoggedError` | if the artifact isn't logged or the run is offline |



#### Examples:

Basic usage
```
# Run logging the artifact
with wandb.init() as r:
 artifact = wandb.Artifact('my_dataset', type='dataset')
 artifact.add_file('path/to/file.txt')
 wandb.log_artifact(artifact)

# Run using the artifact
with wandb.init() as r:
 artifact = r.use_artifact('my_dataset:latest')
 path = artifact.get_path('file.txt')

 # Can now download 'file.txt' directly:
 path.download()
```


### `json_encode`



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/local_artifact.py#L694-L697)

```python
json_encode() -> Dict[str, Any]
```




### `link`



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/artifact.py#L533C5-L545)

```python
link(
 target_path: str,
 aliases: Optional[List[str]] = None
) -> None
```

Link this artifact to a portfolio (a promoted collection of artifacts), with aliases.


| Arguments | |
| :--- | :--- |
| `target_path` | (str) The path to the portfolio. It must take the form {portfolio}, {project}/{portfolio} or {entity}/{project}/{portfolio}. |
| `aliases` | (Optional[List[str]]) A list of strings which uniquely identifies the artifact inside the specified portfolio. |



| Returns | |
| :--- | :--- |
| None |



### `logged_by`



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/local_artifact.py#L343-L347)

```python
logged_by() -> Optional["wandb.apis.public.Run"]
```

Get the run that first logged this artifact.


### `new_file`



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/local_artifact.py#L349-L367)

```python
@contextlib.contextmanager
new_file(
 name: str,
 mode: str = "w",
 encoding: Optional[str] = None
) -> Generator[IO, None, None]
```

Open a new temporary file that will be automatically added to the artifact.


| Arguments | |
| :--- | :--- |
| `name` | (str) The name of the new file being added to the artifact. |
| `mode` | (str, optional) The mode in which to open the new file. |
| `encoding` | (str, optional) The encoding in which to open the new file. |



#### Examples:

```
artifact = wandb.Artifact('my_data', type='dataset')
with artifact.new_file('hello.txt') as f:
 f.write('hello!')
wandb.log_artifact(artifact)
```



| Returns | |
| :--- | :--- |
| (file): A new file object that can be written to. Upon closing, the file will be automatically added to the artifact. |



| Raises | |
| :--- | :--- |
| `ArtifactFinalizedError` | if the artifact has already been finalized. |



### `remove`



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/local_artifact.py#L545-L563)

```python
remove(
 item: Union[str, os.PathLike, ArtifactManifestEntry]
) -> None
```

Remove an item from the artifact.

| Arguments | |
| :--- | :--- |
| `item` | (str, os.PathLike, ArtifactManifestEntry) the item to remove. Can be a specific manifest entry or the name of an artifact-relative path. If the item matches a directory all items in that directory will be removed. |



| Raises | |
| :--- | :--- |
| `ArtifactFinalizedError` | if the artifact has already been finalized. |
| `FileNotFoundError` | if the item isn't found in the artifact. |



| Returns | |
| :--- | :--- |
| None |



### `save`



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/local_artifact.py#L597-L636)

```python
save(
 project: Optional[str] = None,
 settings: Optional['wandb.wandb_sdk.wandb_settings.Settings'] = None
) -> None
```

Persist any changes made to the artifact.

If currently in a run, that run will log this artifact. If not currently in a
run, a run of type "auto" will be created to track this artifact.

| Arguments | |
| :--- | :--- |
| `project` | (str, optional) A project to use for the artifact in the case that a run is not already in context settings: (wandb.Settings, optional) A settings object to use when initializing an automatic run. Most commonly used in testing harness. |



| Returns | |
| :--- | :--- |
| None |



### `used_by`



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/local_artifact.py#L337-L341)

```python
used_by() -> List['wandb.apis.public.Run']
```

Get a list of the runs that have used this artifact.


### `verify`



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/local_artifact.py#L591-L595)

```python
verify(
 root: Optional[str] = None
) -> None
```

Verify that the actual contents of an artifact match the manifest.

All files in the directory are checksummed and the checksums are then
cross-referenced against the artifact's manifest.

NOTE: References are not verified.

| Arguments | |
| :--- | :--- |
| `root` | (str, optional) The directory to verify. If None artifact will be downloaded to './artifacts/self.name/' |



| Raises | |
| :--- | :--- |
| (ValueError): If the verification fails. |



### `wait`



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/local_artifact.py#L644-L653)

```python
wait(
 timeout: Optional[int] = None
) -> ArtifactInterface
```

Wait for an artifact to finish logging.


| Arguments | |
| :--- | :--- |
| `timeout` | (int, optional) Wait up to this long. |



### `__getitem__`



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/artifact.py#L565-L590)

```python
__getitem__(
 name: str
) -> Optional[WBValue]
```

Get the WBValue object located at the artifact relative `name`.


| Arguments | |
| :--- | :--- |
| `name` | (str) The artifact relative name to get |



| Raises | |
| :--- | :--- |
| `ArtifactNotLoggedError` | if the artifact isn't logged or the run is offline |



#### Examples:

Basic usage
```
artifact = wandb.Artifact('my_table', 'dataset')
table = wandb.Table(columns=["a", "b", "c"], data=[[i, i*2, 2**i]])
artifact["my_table"] = table

wandb.log_artifact(artifact)
```

Retrieving an object:
```
artifact = wandb.use_artifact('my_table:latest')
table = artifact["my_table"]
```



### `__setitem__`



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/artifact.py#L592-L621)

```python
__setitem__(
 name: str,
 item: WBValue
) -> ArtifactManifestEntry
```

Add `item` to the artifact at path `name`.


| Arguments | |
| :--- | :--- |
| `name` | (str) The artifact relative name to get |
| `item` | (wandb.WBValue) The object to add. |



| Returns | |
| :--- | :--- |
| `ArtifactManifestEntry` | the added manifest entry |



| Raises | |
| :--- | :--- |
| `ArtifactFinalizedError` | if the artifact has already been finalized. |



#### Examples:

Basic usage
```
artifact = wandb.Artifact('my_table', 'dataset')
table = wandb.Table(columns=["a", "b", "c"], data=[[i, i*2, 2**i]])
artifact["my_table"] = table

wandb.log_artifact(artifact)
```

Retrieving an object:
```
artifact = wandb.use_artifact('my_table:latest')
table = artifact["my_table"]
```
