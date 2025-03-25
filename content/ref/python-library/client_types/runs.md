---
title: runs
object_type: client_type
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/apis/public/runs.py >}}




# <kbd>module</kbd> `wandb.apis.public`
Public API: runs. 



## <kbd>class</kbd> `Runs`
An iterable collection of runs associated with a project and optional filter. 

This is generally used indirectly via the `Api`.runs method. 

### <kbd>method</kbd> `Runs.__init__`

```python
__init__(
    client: 'RetryingClient',
    entity: str,
    project: str,
    filters: Optional[Dict[str, Any]] = None,
    order: Optional[str] = None,
    per_page: int = 50,
    include_sweeps: bool = True
)
```






---

### <kbd>property</kbd> Runs.cursor





---

### <kbd>property</kbd> Runs.length





---

### <kbd>property</kbd> Runs.more







---

### <kbd>method</kbd> `Runs.convert_objects`

```python
convert_objects()
```





---

### <kbd>method</kbd> `Runs.histories`

```python
histories(
    samples: int = 500,
    keys: Optional[List[str]] = None,
    x_axis: str = '_step',
    format: Literal['default', 'pandas', 'polars'] = 'default',
    stream: Literal['default', 'system'] = 'default'
)
```

Return sampled history metrics for all runs that fit the filters conditions. 



**Args:**
 
 - `samples `:  (int, optional) The number of samples to return per run 
 - `keys `:  (list[str], optional) Only return metrics for specific keys 
 - `x_axis `:  (str, optional) Use this metric as the xAxis defaults to _step 
 - `format `:  (Literal, optional) Format to return data in, options are "default", "pandas", "polars" 
 - `stream `:  (Literal, optional) "default" for metrics, "system" for machine metrics 

**Returns:**
 
 - `pandas.DataFrame`:  If format="pandas", returns a `pandas.DataFrame` of history metrics. 
 - `polars.DataFrame`:  If format="polars", returns a `polars.DataFrame` of history metrics. 
 - `list of dicts`:  If format="default", returns a list of dicts containing history metrics with a run_id key. 


---

## <kbd>class</kbd> `Run`
A single run associated with an entity and project. 



**Attributes:**
 
 - `tags` ([str]):  a list of tags associated with the run 
 - `url` (str):  the url of this run 
 - `id` (str):  unique identifier for the run (defaults to eight characters) 
 - `name` (str):  the name of the run 
 - `state` (str):  one of: running, finished, crashed, killed, preempting, preempted 
 - `config` (dict):  a dict of hyperparameters associated with the run 
 - `created_at` (str):  ISO timestamp when the run was started 
 - `system_metrics` (dict):  the latest system metrics recorded for the run 
 - `summary` (dict):  A mutable dict-like property that holds the current summary.  Calling update will persist any changes. 
 - `project` (str):  the project associated with the run 
 - `entity` (str):  the name of the entity associated with the run 
 - `project_internal_id` (int):  the internal id of the project 
 - `user` (str):  the name of the user who created the run 
 - `path` (str):  Unique identifier [entity]/[project]/[run_id] 
 - `notes` (str):  Notes about the run 
 - `read_only` (boolean):  Whether the run is editable 
 - `history_keys` (str):  Keys of the history metrics that have been logged 
 - `with `wandb.log({key`:  value})` 
 - `metadata` (str):  Metadata about the run from wandb-metadata.json 

### <kbd>method</kbd> `Run.__init__`

```python
__init__(
    client: 'RetryingClient',
    entity: str,
    project: str,
    run_id: str,
    attrs: Optional[Mapping] = None,
    include_sweeps: bool = True
)
```

Initialize a Run object. 

Run is always initialized by calling api.runs() where api is an instance of wandb.Api. 


---

### <kbd>property</kbd> Run.entity





---

### <kbd>property</kbd> Run.id





---

### <kbd>property</kbd> Run.json_config





---

### <kbd>property</kbd> Run.lastHistoryStep





---

### <kbd>property</kbd> Run.metadata





---

### <kbd>property</kbd> Run.name





---

### <kbd>property</kbd> Run.path





---

### <kbd>property</kbd> Run.state





---

### <kbd>property</kbd> Run.storage_id





---

### <kbd>property</kbd> Run.summary





---

### <kbd>property</kbd> Run.url





---

### <kbd>property</kbd> Run.username







---

### <kbd>classmethod</kbd> `Run.create`

```python
create(api, run_id=None, project=None, entity=None)
```

Create a run for the given project. 

---

### <kbd>method</kbd> `Run.delete`

```python
delete(delete_artifacts=False)
```

Delete the given run from the wandb backend. 

---

### <kbd>method</kbd> `Run.file`

```python
file(name)
```

Return the path of a file with a given name in the artifact. 



**Args:**
 
 - `name` (str):  name of requested file. 



**Returns:**
 A `File` matching the name argument. 

---

### <kbd>method</kbd> `Run.files`

```python
files(names=None, per_page=50)
```

Return a file path for each file named. 



**Args:**
 
 - `names` (list):  names of the requested files, if empty returns all files 
 - `per_page` (int):  number of results per page. 



**Returns:**
 A `Files` object, which is an iterator over `File` objects. 

---

### <kbd>method</kbd> `Run.history`

```python
history(samples=500, keys=None, x_axis='_step', pandas=True, stream='default')
```

Return sampled history metrics for a run. 

This is simpler and faster if you are ok with the history records being sampled. 



**Args:**
 
 - `samples `:  (int, optional) The number of samples to return 
 - `pandas `:  (bool, optional) Return a pandas dataframe 
 - `keys `:  (list, optional) Only return metrics for specific keys 
 - `x_axis `:  (str, optional) Use this metric as the xAxis defaults to _step 
 - `stream `:  (str, optional) "default" for metrics, "system" for machine metrics 



**Returns:**
 
 - `pandas.DataFrame`:  If pandas=True returns a `pandas.DataFrame` of history  metrics. 
 - `list of dicts`:  If pandas=False returns a list of dicts of history metrics. 

---

### <kbd>method</kbd> `Run.load`

```python
load(force=False)
```





---

### <kbd>method</kbd> `Run.log_artifact`

```python
log_artifact(
    artifact: 'wandb.Artifact',
    aliases: Optional[Collection[str]] = None,
    tags: Optional[Collection[str]] = None
)
```

Declare an artifact as output of a run. 



**Args:**
 
 - `artifact` (`Artifact`):  An artifact returned from  `wandb.Api().artifact(name)`. 
 - `aliases` (list, optional):  Aliases to apply to this artifact. 
 - `tags`:  (list, optional) Tags to apply to this artifact, if any. 



**Returns:**
 A `Artifact` object. 

---

### <kbd>method</kbd> `Run.logged_artifacts`

```python
logged_artifacts(per_page: int = 100) → RunArtifacts
```

Fetches all artifacts logged by this run. 

Retrieves all output artifacts that were logged during the run. Returns a paginated result that can be iterated over or collected into a single list. 



**Args:**
 
 - `per_page`:  Number of artifacts to fetch per API request. 



**Returns:**
 An iterable collection of all Artifact objects logged as outputs during this run. 



**Example:**
 ``` import wandb```
    >>> import tempfile
    >>> with tempfile.NamedTemporaryFile(
    ...     mode="w", delete=False, suffix=".txt"
    ... ) as tmp:
    ...     tmp.write("This is a test artifact")
    ...     tmp_path = tmp.name
    >>> run = wandb.init(project="artifact-example")
    >>> artifact = wandb.Artifact("test_artifact", type="dataset")
    >>> artifact.add_file(tmp_path)
    >>> run.log_artifact(artifact)
    >>> run.finish()
    >>> api = wandb.Api()
    >>> finished_run = api.run(f"{run.entity}/{run.project}/{run.id}")
    >>> for logged_artifact in finished_run.logged_artifacts():
    ...     print(logged_artifact.name)
    test_artifact


---

### <kbd>method</kbd> `Run.save`

```python
save()
```





---

### <kbd>method</kbd> `Run.scan_history`

```python
scan_history(keys=None, page_size=1000, min_step=None, max_step=None)
```

Returns an iterable collection of all history records for a run. 



**Example:**
  Export all the loss values for an example run 

```python
     run = api.run("l2k2/examples-numpy-boston/i0wt6xua")
     history = run.scan_history(keys=["Loss"])
     losses = [row["Loss"] for row in history]
    ``` 



**Args:**
 
 - `keys` ([str], optional):  only fetch these keys, and only fetch rows that have all of keys defined. 
 - `page_size` (int, optional):  size of pages to fetch from the api. 
 - `min_step` (int, optional):  the minimum number of pages to scan at a time. 
 - `max_step` (int, optional):  the maximum number of pages to scan at a time. 



**Returns:**
 An iterable collection over history records (dict). 

---

### <kbd>method</kbd> `Run.to_html`

```python
to_html(height=420, hidden=False)
```

Generate HTML containing an iframe displaying this run. 

---

### <kbd>method</kbd> `Run.update`

```python
update()
```

Persist changes to the run object to the wandb backend. 

---

### <kbd>method</kbd> `Run.upload_file`

```python
upload_file(path, root='.')
```

Upload a file. 



**Args:**
 
 - `path` (str):  name of file to upload. 
 - `root` (str):  the root path to save the file relative to.  i.e.  If you want to have the file saved in the run as "my_dir/file.txt"  and you're currently in "my_dir" you would set root to "../". 



**Returns:**
 A `File` matching the name argument. 

---

### <kbd>method</kbd> `Run.use_artifact`

```python
use_artifact(artifact, use_as=None)
```

Declare an artifact as an input to a run. 



**Args:**
 
 - `artifact` (`Artifact`):  An artifact returned from  `wandb.Api().artifact(name)` 
 - `use_as` (string, optional):  A string identifying  how the artifact is used in the script. Used  to easily differentiate artifacts used in a  run, when using the beta wandb launch  feature's artifact swapping functionality. 



**Returns:**
 A `Artifact` object. 

---

### <kbd>method</kbd> `Run.used_artifacts`

```python
used_artifacts(per_page: int = 100) → RunArtifacts
```

Fetches artifacts explicitly used by this run. 

Retrieves only the input artifacts that were explicitly declared as used during the run, typically via `run.use_artifact()`. Returns a paginated result that can be iterated over or collected into a single list. 



**Args:**
 
 - `per_page`:  Number of artifacts to fetch per API request. 



**Returns:**
 An iterable collection of Artifact objects explicitly used as inputs in this run. 



**Example:**
 ``` import wandb```
    >>> run = wandb.init(project="artifact-example")
    >>> run.use_artifact("test_artifact:latest")
    >>> run.finish()
    >>> api = wandb.Api()
    >>> finished_run = api.run(f"{run.entity}/{run.project}/{run.id}")
    >>> for used_artifact in finished_run.used_artifacts():
    ...     print(used_artifact.name)
    test_artifact


---

### <kbd>method</kbd> `Run.wait_until_finished`

```python
wait_until_finished()
```






