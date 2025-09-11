---
title: Run
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/apis/public/runs.py#L426-L1216 >}}

A single run associated with an entity and project.

| Args |  |
| :--- | :--- |
|  `client` |  The W&B API client. |
|  `entity` |  The entity associated with the run. |
|  `project` |  The project associated with the run. |
|  `run_id` |  The unique identifier for the run. |
|  `attrs` |  The attributes of the run. |
|  `include_sweeps` |  Whether to include sweeps in the run. |

| Attributes |  |
| :--- | :--- |
|  `entity` |  The entity associated with the run. |
|  `id` |  The unique identifier for the run. |
|  `json_config` |  Return the run config as a JSON string. <!-- lazydoc-ignore: internal --> |
|  `lastHistoryStep` |  Returns the last step logged in the run's history. |
|  `metadata` |  Metadata about the run from wandb-metadata.json. Metadata includes the run's description, tags, start time, memory usage and more. |
|  `name` |  The name of the run. |
|  `path` |  The path of the run. The path is a list containing the entity, project, and run_id. |
|  `state` |  The state of the run. Can be one of: Finished, Failed, Crashed, or Running. |
|  `storage_id` |  The unique storage identifier for the run. |
|  `summary` |  A mutable dict-like property that holds summary values associated with the run. |
|  `url` |  The URL of the run. The run URL is generated from the entity, project, and run_id. For SaaS users, it takes the form of `https://wandb.ai/entity/project/run_id`. |
|  `username` |  This API is deprecated. Use `entity` instead. |

## Methods

### `create`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/apis/public/runs.py#L544-L597)

```python
@classmethod
create(
    api: public.Api,
    run_id: (str | None) = None,
    project: (str | None) = None,
    entity: (str | None) = None,
    state: Literal['running', 'pending'] = "running"
)
```

Create a run for the given project.

### `delete`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/apis/public/runs.py#L719-L752)

```python
delete(
    delete_artifacts=(False)
)
```

Delete the given run from the wandb backend.

| Args |  |
| :--- | :--- |
|  delete_artifacts (bool, optional): Whether to delete the artifacts associated with the run. |

### `display`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/apis/attrs.py#L16-L36)

```python
display(
    height=420, hidden=(False)
) -> bool
```

Display this object in jupyter.

### `file`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/apis/public/runs.py#L839-L849)

```python
file(
    name
)
```

Return the path of a file with a given name in the artifact.

| Args |  |
| :--- | :--- |
|  name (str): name of requested file. |

| Returns |  |
| :--- | :--- |
|  A `File` matching the name argument. |

### `files`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/apis/public/runs.py#L808-L837)

```python
files(
    names: (list[str] | None) = None,
    pattern: (str | None) = None,
    per_page: int = 50
)
```

Returns a `Files` object for all files in the run which match the given criteria.

You can specify a list of exact file names to match, or a pattern to match against.
If both are provided, the pattern will be ignored.

| Args |  |
| :--- | :--- |
|  names (list): names of the requested files, if empty returns all files pattern (str, optional): Pattern to match when returning files from W&B. This pattern uses mySQL's LIKE syntax, so matching all files that end with .json would be "%.json". If both names and pattern are provided, a ValueError will be raised. per_page (int): number of results per page. |

| Returns |  |
| :--- | :--- |
|  A `Files` object, which is an iterator over `File` objects. |

### `history`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/apis/public/runs.py#L877-L917)

```python
history(
    samples=500, keys=None, x_axis="_step", pandas=(True), stream="default"
)
```

Return sampled history metrics for a run.

This is simpler and faster if you are ok with the history records being sampled.

| Args |  |
| :--- | :--- |
|  `samples` |  (int, optional) The number of samples to return |
|  `pandas` |  (bool, optional) Return a pandas dataframe |
|  `keys` |  (list, optional) Only return metrics for specific keys |
|  `x_axis` |  (str, optional) Use this metric as the xAxis defaults to _step |
|  `stream` |  (str, optional) "default" for metrics, "system" for machine metrics |

| Returns |  |
| :--- | :--- |
|  `pandas.DataFrame` |  If pandas=True returns a `pandas.DataFrame` of history metrics. list of dicts: If pandas=False returns a list of dicts of history metrics. |

### `load`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/apis/public/runs.py#L599-L638)

```python
load(
    force=(False)
)
```

### `log_artifact`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/apis/public/runs.py#L1082-L1127)

```python
log_artifact(
    artifact: wandb.Artifact,
    aliases: (Collection[str] | None) = None,
    tags: (Collection[str] | None) = None
)
```

Declare an artifact as output of a run.

| Args |  |
| :--- | :--- |
|  artifact (`Artifact`): An artifact returned from `wandb.Api().artifact(name)`. aliases (list, optional): Aliases to apply to this artifact. |
|  `tags` |  (list, optional) Tags to apply to this artifact, if any. |

| Returns |  |
| :--- | :--- |
|  A `Artifact` object. |

### `logged_artifacts`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/apis/public/runs.py#L975-L1011)

```python
logged_artifacts(
    per_page: int = 100
) -> public.RunArtifacts
```

Fetches all artifacts logged by this run.

Retrieves all output artifacts that were logged during the run. Returns a
paginated result that can be iterated over or collected into a single list.

| Args |  |
| :--- | :--- |
|  `per_page` |  Number of artifacts to fetch per API request. |

| Returns |  |
| :--- | :--- |
|  An iterable collection of all Artifact objects logged as outputs during this run. |

#### Example:

```python
import wandb
import tempfile

with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
    tmp.write("This is a test artifact")
    tmp_path = tmp.name
run = wandb.init(project="artifact-example")
artifact = wandb.Artifact("test_artifact", type="dataset")
artifact.add_file(tmp_path)
run.log_artifact(artifact)
run.finish()

api = wandb.Api()

finished_run = api.run(f"{run.entity}/{run.project}/{run.id}")

for logged_artifact in finished_run.logged_artifacts():
    print(logged_artifact.name)
```

### `save`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/apis/public/runs.py#L754-L756)

```python
save()
```

Persist changes to the run object to the W&B backend.

### `scan_history`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/apis/public/runs.py#L919-L973)

```python
scan_history(
    keys=None, page_size=1000, min_step=None, max_step=None
)
```

Returns an iterable collection of all history records for a run.

| Args |  |
| :--- | :--- |
|  keys ([str], optional): only fetch these keys, and only fetch rows that have all of keys defined. page_size (int, optional): size of pages to fetch from the api. min_step (int, optional): the minimum number of pages to scan at a time. max_step (int, optional): the maximum number of pages to scan at a time. |

| Returns |  |
| :--- | :--- |
|  An iterable collection over history records (dict). |

#### Example:

Export all the loss values for an example run

```python
run = api.run("entity/project-name/run-id")
history = run.scan_history(keys=["Loss"])
losses = [row["Loss"] for row in history]
```

### `snake_to_camel`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/apis/attrs.py#L12-L14)

```python
snake_to_camel(
    string
)
```

### `to_html`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/apis/public/runs.py#L1202-L1210)

```python
to_html(
    height=420, hidden=(False)
)
```

Generate HTML containing an iframe displaying this run.

### `update`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/apis/public/runs.py#L691-L717)

```python
update()
```

Persist changes to the run object to the wandb backend.

### `upload_file`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/apis/public/runs.py#L851-L875)

```python
upload_file(
    path, root="."
)
```

Upload a local file to W&B, associating it with this run.

| Args |  |
| :--- | :--- |
|  path (str): Path to the file to upload. Can be absolute or relative. root (str): The root path to save the file relative to. For example, if you want to have the file saved in the run as "my_dir/file.txt" and you're currently in "my_dir" you would set root to "../". Defaults to current directory ("."). |

| Returns |  |
| :--- | :--- |
|  A `File` object representing the uploaded file. |

### `use_artifact`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/apis/public/runs.py#L1044-L1080)

```python
use_artifact(
    artifact, use_as=None
)
```

Declare an artifact as an input to a run.

| Args |  |
| :--- | :--- |
|  artifact (`Artifact`): An artifact returned from `wandb.Api().artifact(name)` use_as (string, optional): A string identifying how the artifact is used in the script. Used to easily differentiate artifacts used in a run, when using the beta wandb launch feature's artifact swapping functionality. |

| Returns |  |
| :--- | :--- |
|  An `Artifact` object. |

### `used_artifacts`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/apis/public/runs.py#L1013-L1042)

```python
used_artifacts(
    per_page: int = 100
) -> public.RunArtifacts
```

Fetches artifacts explicitly used by this run.

Retrieves only the input artifacts that were explicitly declared as used
during the run, typically via `run.use_artifact()`. Returns a paginated
result that can be iterated over or collected into a single list.

| Args |  |
| :--- | :--- |
|  `per_page` |  Number of artifacts to fetch per API request. |

| Returns |  |
| :--- | :--- |
|  An iterable collection of Artifact objects explicitly used as inputs in this run. |

#### Example:

```python
import wandb

run = wandb.init(project="artifact-example")
run.use_artifact("test_artifact:latest")
run.finish()

api = wandb.Api()
finished_run = api.run(f"{run.entity}/{run.project}/{run.id}")
for used_artifact in finished_run.used_artifacts():
    print(used_artifact.name)
test_artifact
```

### `wait_until_finished`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/apis/public/runs.py#L668-L689)

```python
wait_until_finished()
```

Check the state of the run until it is finished.
