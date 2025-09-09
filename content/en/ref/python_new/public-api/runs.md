---
title: runs
object_type: public_apis_namespace
data_type_classification: module
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/runs.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API for Runs. 

This module provides classes for interacting with W&B runs and their associated data. 



**Example:**
 ```python
from wandb.apis.public import Api

# Get runs matching filters
runs = Api().runs(
     path="entity/project", filters={"state": "finished", "config.batch_size": 32}
)

# Access run data
for run in runs:
     print(f"Run: {run.name}")
     print(f"Config: {run.config}")
     print(f"Metrics: {run.summary}")

     # Get history with pandas
     history_df = run.history(keys=["loss", "accuracy"], pandas=True)

     # Work with artifacts
     for artifact in run.logged_artifacts():
         print(f"Artifact: {artifact.name}")
``` 



**Note:**

> This module is part of the W&B Public API and provides read/write access to run data. For logging new runs, use the wandb.init() function from the main wandb package. 

## <kbd>class</kbd> `Runs`
A lazy iterator of `Run` objects associated with a project and optional filter. 

Runs are retrieved in pages from the W&B server as needed. 

This is generally used indirectly using the `Api.runs` namespace. 



**Args:**
 
 - `client`:  (`wandb.apis.public.RetryingClient`) The API client to use  for requests. 
 - `entity`:  (str) The entity (username or team) that owns the project. 
 - `project`:  (str) The name of the project to fetch runs from. 
 - `filters`:  (Optional[Dict[str, Any]]) A dictionary of filters to apply  to the runs query. 
 - `order`:  (str) Order can be `created_at`, `heartbeat_at`, `config.*.value`, or `summary_metrics.*`.  If you prepend order with a + order is ascending (default).  If you prepend order with a - order is descending.  The default order is run.created_at from oldest to newest. 
 - `per_page`:  (int) The number of runs to fetch per request (default is 50). 
 - `include_sweeps`:  (bool) Whether to include sweep information in the  runs. Defaults to True. 



**Examples:**
 ```python
from wandb.apis.public.runs import Runs
from wandb.apis.public import Api

# Get all runs from a project that satisfy the filters
filters = {"state": "finished", "config.optimizer": "adam"}

runs = Api().runs(
    client=api.client,
    entity="entity",
    project="project_name",
    filters=filters,
)

# Iterate over runs and print details
for run in runs:
    print(f"Run name: {run.name}")
    print(f"Run ID: {run.id}")
    print(f"Run URL: {run.url}")
    print(f"Run state: {run.state}")
    print(f"Run config: {run.config}")
    print(f"Run summary: {run.summary}")
    print(f"Run history (samples=5): {run.history(samples=5)}")
    print("----------")

# Get histories for all runs with specific metrics
histories_df = runs.histories(
    samples=100,  # Number of samples per run
    keys=["loss", "accuracy"],  # Metrics to fetch
    x_axis="_step",  # X-axis metric
    format="pandas",  # Return as pandas DataFrame
)
``` 

### <kbd>method</kbd> `Runs.__init__`

```python
__init__(
    client: 'RetryingClient',
    entity: 'str',
    project: 'str',
    filters: 'dict[str, Any] | None' = None,
    order: 'str' = '+created_at',
    per_page: 'int' = 50,
    include_sweeps: 'bool' = True
)
```






---


### <kbd>property</kbd> Runs.length





---



### <kbd>method</kbd> `Runs.histories`

```python
histories(
    samples: 'int' = 500,
    keys: 'list[str] | None' = None,
    x_axis: 'str' = '_step',
    format: "Literal['default', 'pandas', 'polars']" = 'default',
    stream: "Literal['default', 'system']" = 'default'
)
```

Return sampled history metrics for all runs that fit the filters conditions. 



**Args:**
 
 - `samples`:  The number of samples to return per run 
 - `keys`:  Only return metrics for specific keys 
 - `x_axis`:  Use this metric as the xAxis defaults to _step 
 - `format`:  Format to return data in, options are "default", "pandas",  "polars" 
 - `stream`:  "default" for metrics, "system" for machine metrics 

**Returns:**
 
 - `pandas.DataFrame`:  If `format="pandas"`, returns a `pandas.DataFrame`  of history metrics. 
 - `polars.DataFrame`:  If `format="polars"`, returns a `polars.DataFrame`  of history metrics. 
 - `list of dicts`:  If `format="default"`, returns a list of dicts  containing history metrics with a `run_id` key. 


---

## <kbd>class</kbd> `Run`
A single run associated with an entity and project. 



**Args:**
 
 - `client`:  The W&B API client. 
 - `entity`:  The entity associated with the run. 
 - `project`:  The project associated with the run. 
 - `run_id`:  The unique identifier for the run. 
 - `attrs`:  The attributes of the run. 
 - `include_sweeps`:  Whether to include sweeps in the run. 



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
    entity: 'str',
    project: 'str',
    run_id: 'str',
    attrs: 'Mapping | None' = None,
    include_sweeps: 'bool' = True
)
```

Initialize a Run object. 

Run is always initialized by calling api.runs() where api is an instance of wandb.Api. 


---

### <kbd>property</kbd> Run.entity

The entity associated with the run. 

---

### <kbd>property</kbd> Run.id

The unique identifier for the run. 

---


### <kbd>property</kbd> Run.lastHistoryStep

Returns the last step logged in the run's history. 

---

### <kbd>property</kbd> Run.metadata

Metadata about the run from wandb-metadata.json. 

Metadata includes the run's description, tags, start time, memory usage and more. 

---

### <kbd>property</kbd> Run.name

The name of the run. 

---

### <kbd>property</kbd> Run.path

The path of the run. The path is a list containing the entity, project, and run_id. 

---

### <kbd>property</kbd> Run.state

The state of the run. Can be one of: Finished, Failed, Crashed, or Running. 

---

### <kbd>property</kbd> Run.storage_id

The unique storage identifier for the run. 

---

### <kbd>property</kbd> Run.summary

A mutable dict-like property that holds summary values associated with the run. 

---

### <kbd>property</kbd> Run.url

The URL of the run. 

The run URL is generated from the entity, project, and run_id. For SaaS users, it takes the form of `https://wandb.ai/entity/project/run_id`. 

---

### <kbd>property</kbd> Run.username

This API is deprecated. Use `entity` instead. 



---

### <kbd>classmethod</kbd> `Run.create`

```python
create(
    api: 'public.Api',
    run_id: 'str | None' = None,
    project: 'str | None' = None,
    entity: 'str | None' = None,
    state: "Literal['running', 'pending']" = 'running'
)
```

Create a run for the given project. 

---

### <kbd>method</kbd> `Run.delete`

```python
delete(delete_artifacts=False)
```

Delete the given run from the wandb backend. 



**Args:**
 
 - `delete_artifacts` (bool, optional):  Whether to delete the artifacts  associated with the run. 

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
files(
    names: 'list[str] | None' = None,
    pattern: 'str | None' = None,
    per_page: 'int' = 50
)
```

Returns a `Files` object for all files in the run which match the given criteria. 

You can specify a list of exact file names to match, or a pattern to match against. If both are provided, the pattern will be ignored. 



**Args:**
 
 - `names` (list):  names of the requested files, if empty returns all files 
 - `pattern` (str, optional):  Pattern to match when returning files from W&B.  This pattern uses mySQL's LIKE syntax,  so matching all files that end with .json would be "%.json".  If both names and pattern are provided, a ValueError will be raised. 
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
    aliases: 'Collection[str] | None' = None,
    tags: 'Collection[str] | None' = None
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
logged_artifacts(per_page: 'int' = 100) → public.RunArtifacts
```

Fetches all artifacts logged by this run. 

Retrieves all output artifacts that were logged during the run. Returns a paginated result that can be iterated over or collected into a single list. 



**Args:**
 
 - `per_page`:  Number of artifacts to fetch per API request. 



**Returns:**
 An iterable collection of all Artifact objects logged as outputs during this run. 



**Example:**
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

---

### <kbd>method</kbd> `Run.save`

```python
save()
```

Persist changes to the run object to the W&B backend. 

---

### <kbd>method</kbd> `Run.scan_history`

```python
scan_history(keys=None, page_size=1000, min_step=None, max_step=None)
```

Returns an iterable collection of all history records for a run. 



**Args:**
 
 - `keys` ([str], optional):  only fetch these keys, and only fetch rows that have all of keys defined. 
 - `page_size` (int, optional):  size of pages to fetch from the api. 
 - `min_step` (int, optional):  the minimum number of pages to scan at a time. 
 - `max_step` (int, optional):  the maximum number of pages to scan at a time. 



**Returns:**
 An iterable collection over history records (dict). 



**Example:**
 Export all the loss values for an example run 

```python
run = api.run("entity/project-name/run-id")
history = run.scan_history(keys=["Loss"])
losses = [row["Loss"] for row in history]
``` 

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

Upload a local file to W&B, associating it with this run. 



**Args:**
 
 - `path` (str):  Path to the file to upload. Can be absolute or relative. 
 - `root` (str):  The root path to save the file relative to. For example,  if you want to have the file saved in the run as "my_dir/file.txt"  and you're currently in "my_dir" you would set root to "../".  Defaults to current directory ("."). 



**Returns:**
 A `File` object representing the uploaded file. 

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
 An `Artifact` object. 

---

### <kbd>method</kbd> `Run.used_artifacts`

```python
used_artifacts(per_page: 'int' = 100) → public.RunArtifacts
```

Fetches artifacts explicitly used by this run. 

Retrieves only the input artifacts that were explicitly declared as used during the run, typically via `run.use_artifact()`. Returns a paginated result that can be iterated over or collected into a single list. 



**Args:**
 
 - `per_page`:  Number of artifacts to fetch per API request. 



**Returns:**
 An iterable collection of Artifact objects explicitly used as inputs in this run. 



**Example:**
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

---

### <kbd>method</kbd> `Run.wait_until_finished`

```python
wait_until_finished()
```

Check the state of the run until it is finished. 


