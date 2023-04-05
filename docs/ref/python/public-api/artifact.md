# Artifact



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L4214-L5260)



A wandb Artifact.

```python
Artifact(
 client, entity, project, name, attrs=None
)
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
| `entity` | The name of the entity this artifact belongs to. |
| `id` | The artifact's ID. |
| `manifest` | The artifact's manifest. The manifest lists all of its contents, and can't be changed once the artifact has been logged. |
| `metadata` | User-defined artifact metadata. |
| `name` | The artifact's name. |
| `project` | The name of the project this artifact belongs to. |
| `size` | The total size of the artifact in bytes. |
| `source_version` | The artifact's version index under its parent artifact collection. A string with the format "v{number}". |
| `state` | The status of the artifact. One of: "PENDING", "COMMITTED", or "DELETED". |
| `type` | The artifact's type. |
| `updated_at` | The time at which the artifact was last updated. |
| `version` | The artifact's version index under the given artifact collection. A string with the format "v{number}". |



## Methods

### `add`



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L4649-L4650)

```python
add(
 obj, name
)
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



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L4643-L4644)

```python
add_dir(
 path, name=None
)
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



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L4640-L4641)

```python
add_file(
 local_path, name=None, is_tmp=(False)
)
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



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L4646-L4647)

```python
add_reference(
 uri, name=None, checksum=(True), max_objects=None
)
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
- file: The checksum and size will be pulled from the file system. This scheme
 is useful if you have an NFS share or other externally mounted volume
 containing files you wish to track but not necessarily upload.

For any other scheme, the digest is just a hash of the URI and the size is left
blank.

| Arguments | |
| :--- | :--- |
| `uri` | (str) The URI path of the reference to add. Can be an object returned from Artifact.get_path to store a reference to another artifact's entry. |
| `name` | (str) The path within the artifact to place the contents of this reference checksum: (bool, optional) Whether or not to checksum the resource(s) located at the reference URI. Checksumming is strongly recommended as it enables automatic integrity validation, however it can be disabled to speed up artifact creation. (default: True) |
| `max_objects` | (int, optional) The maximum number of objects to consider when adding a reference that points to directory or bucket store prefix. For S3 and GCS, this limit is 10,000 by default but is uncapped for other URI schemes. (default: None) |



| Raises | |
| :--- | :--- |
| `ArtifactFinalizedError` | if the artifact has already been finalized. |



| Returns | |
| :--- | :--- |
| List[ArtifactManifestEntry]: The added manifest entries. |



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



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L4776-L4791)

```python
checkout(
 root=None
)
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



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L4596-L4635)

```python
delete(
 delete_aliases=(False)
)
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



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L4731-L4774)

```python
download(
 root=None, recursive=(False)
)
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



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L4502-L4542)

```python
@staticmethod
expected_type(
 client, name, entity_name, project_name
)
```

Returns the expected type for a given artifact name and project.


### `file`



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L4822-L4842)

```python
file(
 root=None
)
```

Download a single file artifact to dir specified by the root.


| Arguments | |
| :--- | :--- |
| `root` | (str, optional) The root directory in which to place the file. Defaults to './artifacts/self.name/'. |



| Returns | |
| :--- | :--- |
| (str): The full path of the downloaded file. |



### `files`



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L5089-L5100)

```python
files(
 names=None, per_page=50
)
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



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L4298-L4345)

```python
@classmethod
from_id(
 artifact_id: str,
 client: Client
)
```




### `get`



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L4703-L4729)

```python
get(
 name
)
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


### `get_path`



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L4691-L4701)

```python
get_path(
 name
)
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



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L4864-L4865)

```python
json_encode()
```




### `link`



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L4553-L4594)

```python
link(
 target_path, aliases=None
)
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



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L5218-L5254)

```python
logged_by()
```

Retrieve the run which logged this artifact.


| Returns | |
| :--- | :--- |
| `Run` | Run object which logged this artifact |



### `new_file`



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L4637-L4638)

```python
new_file(
 name, mode=None
)
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



### `save`



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L4867-L4930)

```python
save()
```

Persists artifact changes to the wandb backend.


### `used_by`



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L5171-L5216)

```python
used_by()
```

Retrieve the runs which use this artifact directly.


| Returns | |
| :--- | :--- |
| [Run]: a list of Run objects which use this artifact |



### `verify`



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L4793-L4820)

```python
verify(
 root=None
)
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



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L4932-L4933)

```python
wait()
```

Wait for this artifact to finish logging, if needed.


| Returns | |
| :--- | :--- |
| Artifact |



### `__getitem__`



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L5259-L5260)

```python
__getitem__(
 name
)
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






| Class Variables | |
| :--- | :--- |
| `QUERY` | |

