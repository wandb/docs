# Run

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.15.11/wandb/sdk/wandb_run.py#L431-L3729' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


A unit of computation logged by wandb. Typically, this is an ML experiment.

```python
Run(
    settings: Settings,
    config: Optional[Dict[str, Any]] = None,
    sweep_config: Optional[Dict[str, Any]] = None,
    launch_config: Optional[Dict[str, Any]] = None
) -> None
```

Create a run with `wandb.init()`:

<!--yeadoc-test:run-object-basic-->


```python
import wandb

run = wandb.init()
```

There is only ever at most one active `wandb.Run` in any process,
and it is accessible as `wandb.run`:

<!--yeadoc-test:global-run-object-->


```python
import wandb

assert wandb.run is None

wandb.init()

assert wandb.run is not None
```

anything you log with `wandb.log` will be sent to that run.

If you want to start more runs in the same script or notebook, you'll need to
finish the run that is in-flight. Runs can be finished with `wandb.finish` or
by using them in a `with` block:

<!--yeadoc-test:run-context-manager-->


```python
import wandb

wandb.init()
wandb.finish()

assert wandb.run is None

with wandb.init() as run:
    pass  # log data here

assert wandb.run is None
```

See the documentation for `wandb.init` for more on creating runs, or check out
[our guide to `wandb.init`](https://docs.wandb.ai/guides/track/launch).

In distributed training, you can either create a single run in the rank 0 process
and then log information only from that process, or you can create a run in each process,
logging from each separately, and group the results together with the `group` argument
to `wandb.init`. For more details on distributed training with W&B, check out
[our guide](https://docs.wandb.ai/guides/track/log/distributed-training).

Currently, there is a parallel `Run` object in the `wandb.Api`. Eventually these
two objects will be merged.

| Attributes |  |
| :--- | :--- |
|  `summary` |  (Summary) Single values set for each `wandb.log()` key. By default, summary is set to the last value logged. You can manually set summary to the best value, like max accuracy, instead of the final value. |
|  `config` |  Config object associated with this run. |
|  `dir` |  The directory where files associated with the run are saved. |
|  `entity` |  The name of the W&B entity associated with the run. Entity can be a username or the name of a team or organization. |
|  `group` |  Name of the group associated with the run. Setting a group helps the W&B UI organize runs in a sensible way. If you are doing a distributed training you should give all of the runs in the training the same group. If you are doing cross-validation you should give all the cross-validation folds the same group. |
|  `id` |  Identifier for this run. |
|  `mode` |  For compatibility with `0.9.x` and earlier, deprecate eventually. |
|  `name` |  Display name of the run. Display names are not guaranteed to be unique and may be descriptive. By default, they are randomly generated. |
|  `notes` |  Notes associated with the run, if there are any. Notes can be a multiline string and can also use markdown and latex equations inside `$$`, like `$x + 3$`. |
|  `path` |  Path to the run. Run paths include entity, project, and run ID, in the format `entity/project/run_id`. |
|  `project` |  Name of the W&B project associated with the run. |
|  `resumed` |  True if the run was resumed, False otherwise. |
|  `settings` |  A frozen copy of run's Settings object. |
|  `start_time` |  Unix timestamp (in seconds) of when the run started. |
|  `starting_step` |  The first step of the run. |
|  `step` |  Current value of the step. This counter is incremented by `wandb.log`. |
|  `sweep_id` |  ID of the sweep associated with the run, if there is one. |
|  `tags` |  Tags associated with the run, if there are any. |
|  `url` |  The W&B url associated with the run. |

## Methods

### `alert`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.11/wandb/sdk/wandb_run.py#L3072-L3105)

```python
alert(
    title: str,
    text: str,
    level: Optional[Union[str, 'AlertLevel']] = None,
    wait_duration: Union[int, float, timedelta, None] = None
) -> None
```

Launch an alert with the given title and text.

| Arguments |  |
| :--- | :--- |
|  `title` |  (str) The title of the alert, must be less than 64 characters long. |
|  `text` |  (str) The text body of the alert. |
|  `level` |  (str or wandb.AlertLevel, optional) The alert level to use, either: `INFO`, `WARN`, or `ERROR`. |
|  `wait_duration` |  (int, float, or timedelta, optional) The time to wait (in seconds) before sending another alert with this title. |

### `define_metric`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.11/wandb/sdk/wandb_run.py#L2482-L2516)

```python
define_metric(
    name: str,
    step_metric: Union[str, wandb_metric.Metric, None] = None,
    step_sync: Optional[bool] = None,
    hidden: Optional[bool] = None,
    summary: Optional[str] = None,
    goal: Optional[str] = None,
    overwrite: Optional[bool] = None,
    **kwargs
) -> wandb_metric.Metric
```

Define metric properties which will later be logged with `wandb.log()`.

| Arguments |  |
| :--- | :--- |
|  `name` |  Name of the metric. |
|  `step_metric` |  Independent variable associated with the metric. |
|  `step_sync` |  Automatically add `step_metric` to history if needed. Defaults to True if step_metric is specified. |
|  `hidden` |  Hide this metric from automatic plots. |
|  `summary` |  Specify aggregate metrics added to summary. Supported aggregations: "min,max,mean,best,last,none" Default aggregation is `copy` Aggregation `best` defaults to `goal`==`minimize` |
|  `goal` |  Specify direction for optimizing the metric. Supported directions: "minimize,maximize" |

| Returns |  |
| :--- | :--- |
|  A metric object is returned that can be further specified. |

### `detach`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.11/wandb/sdk/wandb_run.py#L2638-L2639)

```python
detach() -> None
```

### `display`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.11/wandb/sdk/wandb_run.py#L1291-L1299)

```python
display(
    height: int = 420,
    hidden: bool = (False)
) -> bool
```

Display this run in jupyter.

### `finish`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.11/wandb/sdk/wandb_run.py#L1911-L1925)

```python
finish(
    exit_code: Optional[int] = None,
    quiet: Optional[bool] = None
) -> None
```

Mark a run as finished, and finish uploading all data.

This is used when creating multiple runs in the same process. We automatically
call this method when your script exits or if you use the run context manager.

| Arguments |  |
| :--- | :--- |
|  `exit_code` |  Set to something other than 0 to mark a run as failed |
|  `quiet` |  Set to true to minimize log output |

### `finish_artifact`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.11/wandb/sdk/wandb_run.py#L2888-L2940)

```python
finish_artifact(
    artifact_or_path: Union[Artifact, str],
    name: Optional[str] = None,
    type: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    distributed_id: Optional[str] = None
) -> Artifact
```

Finishes a non-finalized artifact as output of a run.

Subsequent "upserts" with the same distributed ID will result in a new version.

| Arguments |  |
| :--- | :--- |
|  `artifact_or_path` |  (str or Artifact) A path to the contents of this artifact, can be in the following forms: - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` You can also pass an Artifact object created by calling `wandb.Artifact`. |
|  `name` |  (str, optional) An artifact name. May be prefixed with entity/project. Valid names can be in the following forms: - name:version - name:alias - digest This will default to the basename of the path prepended with the current run id if not specified. |
|  `type` |  (str) The type of artifact to log, examples include `dataset`, `model` |
|  `aliases` |  (list, optional) Aliases to apply to this artifact, defaults to `["latest"]` |
|  `distributed_id` |  (string, optional) Unique string that all distributed jobs share. If None, defaults to the run's group name. |

| Returns |  |
| :--- | :--- |
|  An `Artifact` object. |

### `get_project_url`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.11/wandb/sdk/wandb_run.py#L1173-L1181)

```python
get_project_url() -> Optional[str]
```

Return the url for the W&B project associated with the run, if there is one.

Offline runs will not have a project url.

### `get_sweep_url`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.11/wandb/sdk/wandb_run.py#L1183-L1188)

```python
get_sweep_url() -> Optional[str]
```

Return the url for the sweep associated with the run, if there is one.

### `get_url`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.11/wandb/sdk/wandb_run.py#L1163-L1171)

```python
get_url() -> Optional[str]
```

Return the url for the W&B run, if there is one.

Offline runs will not have a url.

### `join`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.11/wandb/sdk/wandb_run.py#L1959-L1969)

```python
join(
    exit_code: Optional[int] = None
) -> None
```

Deprecated alias for `finish()` - use finish instead.

### `link_artifact`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.11/wandb/sdk/wandb_run.py#L2641-L2687)

```python
link_artifact(
    artifact: Artifact,
    target_path: str,
    aliases: Optional[List[str]] = None
) -> None
```

Link the given artifact to a portfolio (a promoted collection of artifacts).

The linked artifact will be visible in the UI for the specified portfolio.

| Arguments |  |
| :--- | :--- |
|  `artifact` |  the (public or local) artifact which will be linked |
|  `target_path` |  `str` - takes the following forms: {portfolio}, {project}/{portfolio}, or {entity}/{project}/{portfolio} |
|  `aliases` |  `List[str]` - optional alias(es) that will only be applied on this linked artifact inside the portfolio. The alias "latest" will always be applied to the latest version of an artifact that is linked. |

| Returns |  |
| :--- | :--- |
|  None |

### `log`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.11/wandb/sdk/wandb_run.py#L1591-L1792)

```python
log(
    data: Dict[str, Any],
    step: Optional[int] = None,
    commit: Optional[bool] = None,
    sync: Optional[bool] = None
) -> None
```

Log a dictionary of data to the current run's history.

Use `wandb.log` to log data from runs, such as scalars, images, video,
histograms, plots, and tables.

See our [guides to logging](https://docs.wandb.ai/guides/track/log) for
live examples, code snippets, best practices, and more.

The most basic usage is `wandb.log({"train-loss": 0.5, "accuracy": 0.9})`.
This will save the loss and accuracy to the run's history and update
the summary values for these metrics.

Visualize logged data in the workspace at [wandb.ai](https://wandb.ai),
or locally on a [self-hosted instance](https://docs.wandb.ai/guides/hosting)
of the W&B app, or export data to visualize and explore locally, e.g. in
Jupyter notebooks, with [our API](https://docs.wandb.ai/guides/track/public-api-guide).

In the UI, summary values show up in the run table to compare single values across runs.
Summary values can also be set directly with `wandb.run.summary["key"] = value`.

Logged values don't have to be scalars. Logging any wandb object is supported.
For example `wandb.log({"example": wandb.Image("myimage.jpg")})` will log an
example image which will be displayed nicely in the W&B UI.
See the [reference documentation](https://docs.wandb.com/ref/python/data-types)
for all of the different supported types or check out our
[guides to logging](https://docs.wandb.ai/guides/track/log) for examples,
from 3D molecular structures and segmentation masks to PR curves and histograms.
`wandb.Table`s can be used to logged structured data. See our
[guide to logging tables](https://docs.wandb.ai/guides/data-vis/log-tables)
for details.

Logging nested metrics is encouraged and is supported in the W&B UI.
If you log with a nested dictionary like `wandb.log({"train": {"acc": 0.9}, "val": {"acc": 0.8}})`, the metrics will be organized into
`train` and `val` sections in the W&B UI.

wandb keeps track of a global step, which by default increments with each
call to `wandb.log`, so logging related metrics together is encouraged.
If it's inconvenient to log related metrics together
calling `wandb.log({"train-loss": 0.5}, commit=False)` and then
`wandb.log({"accuracy": 0.9})` is equivalent to calling
`wandb.log({"train-loss": 0.5, "accuracy": 0.9})`.

`wandb.log` is not intended to be called more than a few times per second.
If you want to log more frequently than that it's better to aggregate
the data on the client side or you may get degraded performance.

| Arguments |  |
| :--- | :--- |
|  `data` |  (dict, optional) A dict of serializable python objects i.e `str`, `ints`, `floats`, `Tensors`, `dicts`, or any of the `wandb.data_types`. |
|  `commit` |  (boolean, optional) Save the metrics dict to the wandb server and increment the step. If false `wandb.log` just updates the current metrics dict with the data argument and metrics won't be saved until `wandb.log` is called with `commit=True`. |
|  `step` |  (integer, optional) The global step in processing. This persists any non-committed earlier steps but defaults to not committing the specified step. |
|  `sync` |  (boolean, True) This argument is deprecated and currently doesn't change the behaviour of `wandb.log`. |

#### Examples:

For more and more detailed examples, see
[our guides to logging](https://docs.wandb.com/guides/track/log).

### Basic usage

<!--yeadoc-test:init-and-log-basic-->


```python
import wandb

run = wandb.init()
run.log({"accuracy": 0.9, "epoch": 5})
```

### Incremental logging

<!--yeadoc-test:init-and-log-incremental-->


```python
import wandb

run = wandb.init()
run.log({"loss": 0.2}, commit=False)
# Somewhere else when I'm ready to report this step:
run.log({"accuracy": 0.8})
```

### Histogram

<!--yeadoc-test:init-and-log-histogram-->


```python
import numpy as np
import wandb

# sample gradients at random from normal distribution
gradients = np.random.randn(100, 100)
run = wandb.init()
run.log({"gradients": wandb.Histogram(gradients)})
```

### Image from numpy

<!--yeadoc-test:init-and-log-image-numpy-->


```python
import numpy as np
import wandb

run = wandb.init()
examples = []
for i in range(3):
    pixels = np.random.randint(low=0, high=256, size=(100, 100, 3))
    image = wandb.Image(pixels, caption=f"random field {i}")
    examples.append(image)
run.log({"examples": examples})
```

### Image from PIL

<!--yeadoc-test:init-and-log-image-pillow-->


```python
import numpy as np
from PIL import Image as PILImage
import wandb

run = wandb.init()
examples = []
for i in range(3):
    pixels = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
    pil_image = PILImage.fromarray(pixels, mode="RGB")
    image = wandb.Image(pil_image, caption=f"random field {i}")
    examples.append(image)
run.log({"examples": examples})
```

### Video from numpy

<!--yeadoc-test:init-and-log-video-numpy-->


```python
import numpy as np
import wandb

run = wandb.init()
# axes are (time, channel, height, width)
frames = np.random.randint(low=0, high=256, size=(10, 3, 100, 100), dtype=np.uint8)
run.log({"video": wandb.Video(frames, fps=4)})
```

### Matplotlib Plot

<!--yeadoc-test:init-and-log-matplotlib-->


```python
from matplotlib import pyplot as plt
import numpy as np
import wandb

run = wandb.init()
fig, ax = plt.subplots()
x = np.linspace(0, 10)
y = x * x
ax.plot(x, y)  # plot y = x^2
run.log({"chart": fig})
```

### PR Curve

```python
import wandb

run = wandb.init()
run.log({"pr": wandb.plots.precision_recall(y_test, y_probas, labels)})
```

### 3D Object

```python
import wandb

run = wandb.init()
run.log(
    {
        "generated_samples": [
            wandb.Object3D(open("sample.obj")),
            wandb.Object3D(open("sample.gltf")),
            wandb.Object3D(open("sample.glb")),
        ]
    }
)
```

| Raises |  |
| :--- | :--- |
|  `wandb.Error` |  if called before `wandb.init` |
|  `ValueError` |  if invalid data is passed |

### `log_artifact`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.11/wandb/sdk/wandb_run.py#L2797-L2832)

```python
log_artifact(
    artifact_or_path: Union[Artifact, StrPath],
    name: Optional[str] = None,
    type: Optional[str] = None,
    aliases: Optional[List[str]] = None
) -> Artifact
```

Declare an artifact as an output of a run.

| Arguments |  |
| :--- | :--- |
|  `artifact_or_path` |  (str or Artifact) A path to the contents of this artifact, can be in the following forms: - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` You can also pass an Artifact object created by calling `wandb.Artifact`. |
|  `name` |  (str, optional) An artifact name. May be prefixed with entity/project. Valid names can be in the following forms: - name:version - name:alias - digest This will default to the basename of the path prepended with the current run id if not specified. |
|  `type` |  (str) The type of artifact to log, examples include `dataset`, `model` |
|  `aliases` |  (list, optional) Aliases to apply to this artifact, defaults to `["latest"]` |

| Returns |  |
| :--- | :--- |
|  An `Artifact` object. |

### `log_code`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.11/wandb/sdk/wandb_run.py#L1081-L1161)

```python
log_code(
    root: Optional[str] = ".",
    name: Optional[str] = None,
    include_fn: Union[Callable[[str, str], bool], Callable[[str], bool]] = _is_py_or_dockerfile,
    exclude_fn: Union[Callable[[str, str], bool], Callable[[str], bool]] = filenames.exclude_wandb_fn
) -> Optional[Artifact]
```

Save the current state of your code to a W&B Artifact.

By default, it walks the current directory and logs all files that end with `.py`.

| Arguments |  |
| :--- | :--- |
|  `root` |  The relative (to `os.getcwd()`) or absolute path to recursively find code from. |
|  `name` |  (str, optional) The name of our code artifact. By default, we'll name the artifact `source-$PROJECT_ID-$ENTRYPOINT_RELPATH`. There may be scenarios where you want many runs to share the same artifact. Specifying name allows you to achieve that. |
|  `include_fn` |  A callable that accepts a file path and (optionally) root path and returns True when it should be included and False otherwise. This defaults to: `lambda path, root: path.endswith(".py")` |
|  `exclude_fn` |  A callable that accepts a file path and (optionally) root path and returns `True` when it should be excluded and `False` otherwise. This defaults to a function that excludes all files within `&lt;root&gt;/.wandb/` and `&lt;root&gt;/wandb/` directories. |

#### Examples:

Basic usage

```python
run.log_code()
```

Advanced usage

```python
run.log_code(
    "../",
    include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"),
    exclude_fn=lambda path, root: os.path.relpath(path, root).startswith("cache/"),
)
```

| Returns |  |
| :--- | :--- |
|  An `Artifact` object if code was logged |

### `mark_preempting`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.11/wandb/sdk/wandb_run.py#L3123-L3131)

```python
mark_preempting() -> None
```

Mark this run as preempting.

Also tells the internal process to immediately report this to server.

### `plot_table`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.11/wandb/sdk/wandb_run.py#L1996-L2014)

```python
@staticmethod
plot_table(
    vega_spec_name: str,
    data_table: "wandb.Table",
    fields: Dict[str, Any],
    string_fields: Optional[Dict[str, Any]] = None
) -> CustomChart
```

Create a custom plot on a table.

| Arguments |  |
| :--- | :--- |
|  `vega_spec_name` |  the name of the spec for the plot |
|  `data_table` |  a wandb.Table object containing the data to be used on the visualization |
|  `fields` |  a dict mapping from table keys to fields that the custom visualization needs |
|  `string_fields` |  a dict that provides values for any string constants the custom visualization needs |

### `project_name`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.11/wandb/sdk/wandb_run.py#L1027-L1028)

```python
project_name() -> str
```

### `restore`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.11/wandb/sdk/wandb_run.py#L1896-L1909)

```python
restore(
    name: str,
    run_path: Optional[str] = None,
    replace: bool = (False),
    root: Optional[str] = None
) -> Union[None, TextIO]
```

Download the specified file from cloud storage.

File is placed into the current directory or run directory.
By default, will only download the file if it doesn't already exist.

| Arguments |  |
| :--- | :--- |
|  `name` |  the name of the file |
|  `run_path` |  optional path to a run to pull files from, i.e. `username/project_name/run_id` if wandb.init has not been called, this is required. |
|  `replace` |  whether to download the file even if it already exists locally |
|  `root` |  the directory to download the file to. Defaults to the current directory or the run directory if wandb.init was called. |

| Returns |  |
| :--- | :--- |
|  None if it can't find the file, otherwise a file object open for reading |

| Raises |  |
| :--- | :--- |
|  `wandb.CommError` |  if we can't connect to the wandb backend |
|  `ValueError` |  if the file is not found or can't find run_path |

### `save`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.11/wandb/sdk/wandb_run.py#L1794-L1824)

```python
save(
    glob_str: Optional[str] = None,
    base_path: Optional[str] = None,
    policy: "PolicyName" = "live"
) -> Union[bool, List[str]]
```

Ensure all files matching `glob_str` are synced to wandb with the policy specified.

| Arguments |  |
| :--- | :--- |
|  `glob_str` |  (string) a relative or absolute path to a unix glob or regular path. If this isn't specified the method is a noop. |
|  `base_path` |  (string) the base path to run the glob relative to |
|  `policy` |  (string) on of `live`, `now`, or `end` - live: upload the file as it changes, overwriting the previous version - now: upload the file once now - end: only upload file when the run ends |

### `status`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.11/wandb/sdk/wandb_run.py#L1971-L1994)

```python
status() -> RunStatus
```

Get sync info from the internal backend, about the current run's sync status.

### `to_html`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.11/wandb/sdk/wandb_run.py#L1301-L1310)

```python
to_html(
    height: int = 420,
    hidden: bool = (False)
) -> str
```

Generate HTML containing an iframe displaying the current run.

### `unwatch`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.11/wandb/sdk/wandb_run.py#L2599-L2601)

```python
unwatch(
    models=None
) -> None
```

### `upsert_artifact`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.11/wandb/sdk/wandb_run.py#L2834-L2886)

```python
upsert_artifact(
    artifact_or_path: Union[Artifact, str],
    name: Optional[str] = None,
    type: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    distributed_id: Optional[str] = None
) -> Artifact
```

Declare (or append to) a non-finalized artifact as output of a run.

Note that you must call run.finish_artifact() to finalize the artifact.
This is useful when distributed jobs need to all contribute to the same artifact.

| Arguments |  |
| :--- | :--- |
|  `artifact_or_path` |  (str or Artifact) A path to the contents of this artifact, can be in the following forms: - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` You can also pass an Artifact object created by calling `wandb.Artifact`. |
|  `name` |  (str, optional) An artifact name. May be prefixed with entity/project. Valid names can be in the following forms: - name:version - name:alias - digest This will default to the basename of the path prepended with the current run id if not specified. |
|  `type` |  (str) The type of artifact to log, examples include `dataset`, `model` |
|  `aliases` |  (list, optional) Aliases to apply to this artifact, defaults to `["latest"]` |
|  `distributed_id` |  (string, optional) Unique string that all distributed jobs share. If None, defaults to the run's group name. |

| Returns |  |
| :--- | :--- |
|  An `Artifact` object. |

### `use_artifact`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.11/wandb/sdk/wandb_run.py#L2689-L2795)

```python
use_artifact(
    artifact_or_name: Union[str, Artifact],
    type: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    use_as: Optional[str] = None
) -> Artifact
```

Declare an artifact as an input to a run.

Call `download` or `file` on the returned object to get the contents locally.

| Arguments |  |
| :--- | :--- |
|  `artifact_or_name` |  (str or Artifact) An artifact name. May be prefixed with entity/project/. Valid names can be in the following forms: - name:version - name:alias - digest You can also pass an Artifact object created by calling `wandb.Artifact` |
|  `type` |  (str, optional) The type of artifact to use. |
|  `aliases` |  (list, optional) Aliases to apply to this artifact |
|  `use_as` |  (string, optional) Optional string indicating what purpose the artifact was used with. Will be shown in UI. |

| Returns |  |
| :--- | :--- |
|  An `Artifact` object. |

### `watch`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.11/wandb/sdk/wandb_run.py#L2586-L2596)

```python
watch(
    models, criterion=None, log="gradients", log_freq=100, idx=None,
    log_graph=(False)
) -> None
```

### `__enter__`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.11/wandb/sdk/wandb_run.py#L3107-L3108)

```python
__enter__() -> "Run"
```

### `__exit__`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.11/wandb/sdk/wandb_run.py#L3110-L3121)

```python
__exit__(
    exc_type: Type[BaseException],
    exc_val: BaseException,
    exc_tb: TracebackType
) -> bool
```
