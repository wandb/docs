# Artifact



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/public_artifact.py#L90-L1188)



A wandb Artifact.

```python
Artifact(
 client: RetryingClient,
 entity: str,
 project: str,
 name: str,
 attrs: Optional[Dict[str, Any]] = None
) -> None
```




An artifact that has been logged, including all its attributes, links to the runs
that use it, and a link to the run that logged it.

#### Examples:

Basic usage
```
api = wandb.Api()
artifact = api.artifact('project/artifact:alias')

# Get information about the artifact...
artifact.digest
artifact.aliases
```

Updating an artifact
```
artifact = api.artifact('project/artifact:alias')

# Update the description
artifact.description = 'My new description'

# Selectively update metadata keys
artifact.metadata["oldKey"] = "new value"

# Replace the metadata entirely
artifact.metadata = {"newKey": "new value"}

# Add an alias
artifact.aliases.append('best')

# Remove an alias
artifact.aliases.remove('latest')

# Completely replace the aliases
artifact.aliases = ['replaced']

# Persist all artifact modifications
artifact.save()
```

Artifact graph traversal
```
artifact = api.artifact('project/artifact:alias')

# Walk up and down the graph from an artifact:
producer_run = artifact.logged_by()
consumer_runs = artifact.used_by()

# Walk up and down the graph from a run:
logged_artifacts = run.logged_artifacts()
used_artifacts = run.used_artifacts()
```

Deleting an artifact
```
artifact = api.artifact('project/artifact:alias')
artifact.delete()
```




| Attributes | |
| :--- | :--- |
| `aliases` | The aliases associated with this artifact. |
| `commit_hash` | The hash returned when this artifact was committed. |
| `created_at` | The time at which the artifact was created. |
| `description` | The artifact description. |
| `digest` | The logical digest of the artifact. The digest is the checksum of the artifact's contents. If an artifact has the same digest as the current `latest` version, then `log_artifact` is a no-op. |
| `entity` | The name of the entity of the secondary (portfolio) artifact collection. |
| `id` | The artifact's ID. |
| `manifest` | The artifact's manifest. The manifest lists all of its contents, and can't be changed once the artifact has been logged. |
| `metadata` | User-defined artifact metadata. |
| `name` | The artifact name and version in its secondary (portfolio) collection. A string with the format {collection}:{alias}. Before the artifact is saved,contains only the name since the version is not yet known. |
| `project` | The name of the project of the secondary (portfolio) artifact collection. |
| `qualified_name` | The entity/project/name of the secondary (portfolio) collection. |
| `size` | The total size of the artifact in bytes. |
| `source_entity` | The name of the entity of the primary (sequence) artifact collection. |
| `source_name` | The artifact name and version in its primary (sequence) collection. A string with the format {collection}:{alias}. Before the artifact is saved,contains only the name since the version is not yet known. |
| `source_project` | The name of the project of the primary (sequence) artifact collection. |
| `source_qualified_name` | The entity/project/name of the primary (sequence) collection. |
| `source_version` | The artifact's version in its primary (sequence) collection. A string with the format "v{number}". |
| `state` | The status of the artifact. One of: "PENDING", "COMMITTED", or "DELETED". |
| `type` | The artifact's type. |
| `updated_at` | The time at which the artifact was last updated. |
| `version` | The artifact's version in its secondary (portfolio) collection. A string with the format "v{number}". |



## Methods

### `add`



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/public_artifact.py#L544-L545)

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



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/public_artifact.py#L532-L533)

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



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/public_artifact.py#L524-L530)

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



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/public_artifact.py#L535-L542)

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
| `checksum`: (bool, optional) Whether or not to checksum the resource(s) located at the reference URI. Checksumming is strongly recommended as it enables automatic integrity validation, however it can be disabled to speed up artifact creation. (default: True) |
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



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/public_artifact.py#L698-L711)

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



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/public_artifact.py#L478-L516)

```python
delete(
 delete_aliases: bool = False
) -> None
```

Delete an artifact and its files.


#### Examples:

Delete all the "model" artifacts a run has logged:
```
runs = api.runs(path="my_entity/my_project")
for run in runs:
 for artifact in run.logged_artifacts():
 if artifact.type == "model":
 artifact.delete(delete_aliases=True)
```



| Arguments | |
| :--- | :--- |
| `delete_aliases` | (bool) If true, deletes all aliases associated with the artifact. Otherwise, this raises an exception if the artifact has existing aliases. |



### `download`



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/public_artifact.py#L629-L696)

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



### `expected_type`



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/public_artifact.py#L384-L426)

```python
@staticmethod
expected_type(
 client: RetryingClient,
 name: str,
 entity_name: str,
 project_name: str
) -> Optional[str]
```

Returns the expected type for a given artifact name and project.


### `file`



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/public_artifact.py#L740-L760)

```python
file(
 root: Optional[str] = None
) -> StrPath
```

Download a single file artifact to dir specified by the root.


| Arguments | |
| :--- | :--- |
| `root` | (str, optional) The root directory in which to place the file. Defaults to './artifacts/self.name/'. |



| Returns | |
| :--- | :--- |
| (str): The full path of the downloaded file. |



### `files`



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/public_artifact.py#L1003-L1016)

```python
files(
 names: Optional[List[str]] = None,
 per_page: int = 50
) -> ArtifactFiles
```

Iterate over all files stored in this artifact.


| Arguments | |
| :--- | :--- |
| `names` | (list of str, optional) The filename paths relative to the root of the artifact you wish to list. |
| `per_page` | (int, default 50) The number of files to return per request |



| Returns | |
| :--- | :--- |
| (`ArtifactFiles`): An iterator containing `File` objects |



### `from_id`



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/public_artifact.py#L174-L210)

```python
@classmethod
from_id(
 artifact_id: str,
 client: RetryingClient
) -> Optional[Artifact]
```



### `get`



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/public_artifact.py#L600-L627)

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
 table = artifact.get('my_table')
```


### `get_path`



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/public_artifact.py#L591-L598)

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



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/public_artifact.py#L782-L783)

```python
json_encode() -> Dict[str, Any]
```




### `link`



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/public_artifact.py#L436-L476)

```python
@normalize_exceptions
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



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/public_artifact.py#L1138-L1175)

```python
logged_by() -> Optional[Run]
```

Retrieve the run which logged this artifact.


| Returns | |
| :--- | :--- |
| `Run` | Run object which logged this artifact |



### `new_draft`



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/public_artifact.py#L1177-L1188)

```python
new_draft() -> LocalArtifact
```

Create a new draft artifact with the same content as this committed artifact. The artifact returned can be extended or modified and logged as a new version.



### `new_file`



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/public_artifact.py#L518-L522)

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



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/public_artifact.py#L547-L548)

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



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/public_artifact.py#L785-L847)

```python
save() -> None
```

Persists artifact changes to the wandb backend.


### `used_by`



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/public_artifact.py#L1091-L1136)

```python
used_by() -> List[Run]
```

Retrieve the runs which use this artifact directly.


| Returns | |
| :--- | :--- |
| [Run]: a list of Run objects which use this artifact |



### `verify`



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/public_artifact.py#L713-L738)

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



[View source](https://github.com/wandb/wandb/blob/v0.15.4/wandb/sdk/artifacts/public_artifact.py#L849-L850)

```python
wait() -> Artifact
```

Wait for this artifact to finish logging, if needed.


| Returns | |
| :--- | :--- |
| Artifact |



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
| `name` | (str) The path within the artifact to add the object. |
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



| Class Variables | |
| :--- | :--- |
| `QUERY` | |

