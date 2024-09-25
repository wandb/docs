# Run

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L465-L4299' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


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

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L3594-L3627)

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
|  `level` |  (str or AlertLevel, optional) The alert level to use, either: `INFO`, `WARN`, or `ERROR`. |
|  `wait_duration` |  (int, float, or timedelta, optional) The time to wait (in seconds) before sending another alert with this title. |

### `define_metric`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L2746-L2807)

```python
define_metric(
    name: str,
    step_metric: Union[str, wandb_metric.Metric, None] = None,
    step_sync: Optional[bool] = None,
    hidden: Optional[bool] = None,
    summary: Optional[str] = None,
    goal: Optional[str] = None,
    overwrite: Optional[bool] = None
) -> wandb_metric.Metric
```

Customize metrics logged with `wandb.log()`.

| Arguments |  |
| :--- | :--- |
|  `name` |  The name of the metric to customize. |
|  `step_metric` |  The name of another metric to serve as the X-axis for this metric in automatically generated charts. |
|  `step_sync` |  Automatically insert the last value of step_metric into `run.log()` if it is not provided explicitly. Defaults to True if step_metric is specified. |
|  `hidden` |  Hide this metric from automatic plots. |
|  `summary` |  Specify aggregate metrics added to summary. Supported aggregations include "min", "max", "mean", "last", "best", "copy" and "none". "best" is used together with the goal parameter. "none" prevents a summary from being generated. "copy" is deprecated and should not be used. |
|  `goal` |  Specify how to interpret the "best" summary type. Supported options are "minimize" and "maximize". |
|  `overwrite` |  If false, then this call is merged with previous `define_metric` calls for the same metric by using their values for any unspecified parameters. If true, then unspecified parameters overwrite values specified by previous calls. |

| Returns |  |
| :--- | :--- |
|  An object that represents this call but can otherwise be discarded. |

### `detach`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L2940-L2941)

```python
detach() -> None
```

### `display`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L1360-L1368)

```python
display(
    height: int = 420,
    hidden: bool = (False)
) -> bool
```

Display this run in jupyter.

### `finish`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L2142-L2156)

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

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L3203-L3255)

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

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L1242-L1250)

```python
get_project_url() -> Optional[str]
```

Return the url for the W&B project associated with the run, if there is one.

Offline runs will not have a project url.

### `get_sweep_url`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L1252-L1257)

```python
get_sweep_url() -> Optional[str]
```

Return the url for the sweep associated with the run, if there is one.

### `get_url`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L1232-L1240)

```python
get_url() -> Optional[str]
```

Return the url for the W&B run, if there is one.

Offline runs will not have a url.

### `join`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L2210-L2221)

```python
join(
    exit_code: Optional[int] = None
) -> None
```

Deprecated alias for `finish()` - use finish instead.

### `link_artifact`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L2943-L2996)

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

### `link_model`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L3500-L3592)

```python
link_model(
    path: StrPath,
    registered_model_name: str,
    name: Optional[str] = None,
    aliases: Optional[List[str]] = None
) -> None
```

Log a model artifact version and link it to a registered model in the model registry.

The linked model version will be visible in the UI for the specified registered model.

#### Steps:

- Check if 'name' model artifact has been logged. If so, use the artifact version that matches the files
  located at 'path' or log a new version. Otherwise log files under 'path' as a new model artifact, 'name'
  of type 'model'.
- Check if registered model with name 'registered_model_name' exists in the 'model-registry' project.
  If not, create a new registered model with name 'registered_model_name'.
- Link version of model artifact 'name' to registered model, 'registered_model_name'.
- Attach aliases from 'aliases' list to the newly linked model artifact version.

| Arguments |  |
| :--- | :--- |
|  `path` |  (str) A path to the contents of this model, can be in the following forms: - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` |
|  `registered_model_name` |  (str) - the name of the registered model that the model is to be linked to. A registered model is a collection of model versions linked to the model registry, typically representing a team's specific ML Task. The entity that this registered model belongs to will be derived from the run |
|  `name` |  (str, optional) - the name of the model artifact that files in 'path' will be logged to. This will default to the basename of the path prepended with the current run id if not specified. |
|  `aliases` |  (List[str], optional) - alias(es) that will only be applied on this linked artifact inside the registered model. The alias "latest" will always be applied to the latest version of an artifact that is linked. |

#### Examples:

```python
run.link_model(
    path="/local/directory",
    registered_model_name="my_reg_model",
    name="my_model_artifact",
    aliases=["production"],
)
```

Invalid usage

```python
run.link_model(
    path="/local/directory",
    registered_model_name="my_entity/my_project/my_reg_model",
    name="my_model_artifact",
    aliases=["production"],
)

run.link_model(
    path="/local/directory",
    registered_model_name="my_reg_model",
    name="my_entity/my_project/my_model_artifact",
    aliases=["production"],
)
```

| Raises |  |
| :--- | :--- |
|  `AssertionError` |  if registered_model_name is a path or if model artifact 'name' is of a type that does not contain the substring 'model' |
|  `ValueError` |  if name has invalid special characters |

| Returns |  |
| :--- | :--- |
|  None |

### `log`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L1678-L1933)

```python
log(
    data: Dict[str, Any],
    step: Optional[int] = None,
    commit: Optional[bool] = None,
    sync: Optional[bool] = None
) -> None
```

Upload run data.

Use `log` to log data from runs, such as scalars, images, video,
histograms, plots, and tables.

See our [guides to logging](https://docs.wandb.ai/guides/track/log) for
live examples, code snippets, best practices, and more.

The most basic usage is `run.log({"train-loss": 0.5, "accuracy": 0.9})`.
This will save the loss and accuracy to the run's history and update
the summary values for these metrics.

Visualize logged data in the workspace at [wandb.ai](https://wandb.ai),
or locally on a [self-hosted instance](https://docs.wandb.ai/guides/hosting)
of the W&B app, or export data to visualize and explore locally, e.g. in
Jupyter notebooks, with [our API](https://docs.wandb.ai/guides/track/public-api-guide).

Logged values don't have to be scalars. Logging any wandb object is supported.
For example `run.log({"example": wandb.Image("myimage.jpg")})` will log an
example image which will be displayed nicely in the W&B UI.
See the [reference documentation](https://docs.wandb.com/ref/python/data-types)
for all of the different supported types or check out our
[guides to logging](https://docs.wandb.ai/guides/track/log) for examples,
from 3D molecular structures and segmentation masks to PR curves and histograms.
You can use `wandb.Table` to log structured data. See our
[guide to logging tables](https://docs.wandb.ai/guides/data-vis/log-tables)
for details.

The W&B UI organizes metrics with a forward slash (`/`) in their name
into sections named using the text before the final slash. For example,
the following results in two sections named "train" and "validate":

```
run.log({
    "train/accuracy": 0.9,
    "train/loss": 30,
    "validate/accuracy": 0.8,
    "validate/loss": 20,
})
```

Only one level of nesting is supported; `run.log({"a/b/c": 1})`
produces a section named "a/b".

`run.log` is not intended to be called more than a few times per second.
For optimal performance, limit your logging to once every N iterations,
or collect data over multiple iterations and log it in a single step.

### The W&B step

With basic usage, each call to `log` creates a new "step".
The step must always increase, and it is not possible to log
to a previous step.

Note that you can use any metric as the X axis in charts.
In many cases, it is better to treat the W&B step like
you'd treat a timestamp rather than a training step.

```
# Example: log an "epoch" metric for use as an X axis.
run.log({"epoch": 40, "train-loss": 0.5})
```

See also [define_metric](https://docs.wandb.ai/ref/python/run#define_metric).

It is possible to use multiple `log` invocations to log to
the same step with the `step` and `commit` parameters.
The following are all equivalent:

```
# Normal usage:
run.log({"train-loss": 0.5, "accuracy": 0.8})
run.log({"train-loss": 0.4, "accuracy": 0.9})

# Implicit step without auto-incrementing:
run.log({"train-loss": 0.5}, commit=False)
run.log({"accuracy": 0.8})
run.log({"train-loss": 0.4}, commit=False)
run.log({"accuracy": 0.9})

# Explicit step:
run.log({"train-loss": 0.5}, step=current_step)
run.log({"accuracy": 0.8}, step=current_step)
current_step += 1
run.log({"train-loss": 0.4}, step=current_step)
run.log({"accuracy": 0.9}, step=current_step)
```

| Arguments |  |
| :--- | :--- |
|  `data` |  A `dict` with `str` keys and values that are serializable Python objects including: `int`, `float` and `string`; any of the `wandb.data_types`; lists, tuples and NumPy arrays of serializable Python objects; other `dict`s of this structure. |
|  `step` |  The step number to log. If `None`, then an implicit auto-incrementing step is used. See the notes in the description. |
|  `commit` |  If true, finalize and upload the step. If false, then accumulate data for the step. See the notes in the description. If `step` is `None`, then the default is `commit=True`; otherwise, the default is `commit=False`. |
|  `sync` |  This argument is deprecated and does nothing. |

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
run.log({"pr": wandb.plot.pr_curve(y_test, y_probas, labels)})
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

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L3107-L3147)

```python
log_artifact(
    artifact_or_path: Union[Artifact, StrPath],
    name: Optional[str] = None,
    type: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    tags: Optional[List[str]] = None
) -> Artifact
```

Declare an artifact as an output of a run.

| Arguments |  |
| :--- | :--- |
|  `artifact_or_path` |  (str or Artifact) A path to the contents of this artifact, can be in the following forms: - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` You can also pass an Artifact object created by calling `wandb.Artifact`. |
|  `name` |  (str, optional) An artifact name. Valid names can be in the following forms: - name:version - name:alias - digest This will default to the basename of the path prepended with the current run id if not specified. |
|  `type` |  (str) The type of artifact to log, examples include `dataset`, `model` |
|  `aliases` |  (list, optional) Aliases to apply to this artifact, defaults to `["latest"]` |
|  `tags` |  (list, optional) Tags to apply to this artifact, if any. |

| Returns |  |
| :--- | :--- |
|  An `Artifact` object. |

### `log_code`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L1147-L1230)

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

### `log_model`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L3396-L3445)

```python
log_model(
    path: StrPath,
    name: Optional[str] = None,
    aliases: Optional[List[str]] = None
) -> None
```

Logs a model artifact containing the contents inside the 'path' to a run and marks it as an output to this run.

| Arguments |  |
| :--- | :--- |
|  `path` |  (str) A path to the contents of this model, can be in the following forms: - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` |
|  `name` |  (str, optional) A name to assign to the model artifact that the file contents will be added to. The string must contain only the following alphanumeric characters: dashes, underscores, and dots. This will default to the basename of the path prepended with the current run id if not specified. |
|  `aliases` |  (list, optional) Aliases to apply to the created model artifact, defaults to `["latest"]` |

#### Examples:

```python
run.log_model(
    path="/local/directory",
    name="my_model_artifact",
    aliases=["production"],
)
```

Invalid usage

```python
run.log_model(
    path="/local/directory",
    name="my_entity/my_project/my_model_artifact",
    aliases=["production"],
)
```

| Raises |  |
| :--- | :--- |
|  `ValueError` |  if name has invalid special characters |

| Returns |  |
| :--- | :--- |
|  None |

### `mark_preempting`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L3645-L3653)

```python
mark_preempting() -> None
```

Mark this run as preempting.

Also tells the internal process to immediately report this to server.

### `plot_table`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L2248-L2269)

```python
@staticmethod
plot_table(
    vega_spec_name: str,
    data_table: "wandb.Table",
    fields: Dict[str, Any],
    string_fields: Optional[Dict[str, Any]] = None,
    split_table: Optional[bool] = (False)
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

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L1092-L1093)

```python
project_name() -> str
```

### `restore`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L2127-L2140)

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

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L1935-L2041)

```python
save(
    glob_str: Optional[Union[str, os.PathLike]] = None,
    base_path: Optional[Union[str, os.PathLike]] = None,
    policy: PolicyName = "live"
) -> Union[bool, List[str]]
```

Sync one or more files to W&B.

Relative paths are relative to the current working directory.

A Unix glob, such as "myfiles/*", is expanded at the time `save` is
called regardless of the `policy`. In particular, new files are not
picked up automatically.

A `base_path` may be provided to control the directory structure of
uploaded files. It should be a prefix of `glob_str`, and the directory
structure beneath it is preserved. It's best understood through
examples:

```
wandb.save("these/are/myfiles/*")
# => Saves files in a "these/are/myfiles/" folder in the run.

wandb.save("these/are/myfiles/*", base_path="these")
# => Saves files in an "are/myfiles/" folder in the run.

wandb.save("/User/username/Documents/run123/*.txt")
# => Saves files in a "run123/" folder in the run. See note below.

wandb.save("/User/username/Documents/run123/*.txt", base_path="/User")
# => Saves files in a "username/Documents/run123/" folder in the run.

wandb.save("files/*/saveme.txt")
# => Saves each "saveme.txt" file in an appropriate subdirectory
#    of "files/".
```

Note: when given an absolute path or glob and no `base_path`, one
directory level is preserved as in the example above.

| Arguments |  |
| :--- | :--- |
|  `glob_str` |  A relative or absolute path or Unix glob. |
|  `base_path` |  A path to use to infer a directory structure; see examples. |
|  `policy` |  One of `live`, `now`, or `end`. * live: upload the file as it changes, overwriting the previous version * now: upload the file once now * end: upload file when the run ends |

| Returns |  |
| :--- | :--- |
|  Paths to the symlinks created for the matched files. For historical reasons, this may return a boolean in legacy code. |

### `status`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L2223-L2246)

```python
status() -> RunStatus
```

Get sync info from the internal backend, about the current run's sync status.

### `to_html`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L1370-L1379)

```python
to_html(
    height: int = 420,
    hidden: bool = (False)
) -> str
```

Generate HTML containing an iframe displaying the current run.

### `unwatch`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L2901-L2903)

```python
unwatch(
    models=None
) -> None
```

### `upsert_artifact`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L3149-L3201)

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

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L2998-L3105)

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
|  `artifact_or_name` |  (str or Artifact) An artifact name. May be prefixed with entity/project/. Valid names can be in the following forms: - name:version - name:alias You can also pass an Artifact object created by calling `wandb.Artifact` |
|  `type` |  (str, optional) The type of artifact to use. |
|  `aliases` |  (list, optional) Aliases to apply to this artifact |
|  `use_as` |  (string, optional) Optional string indicating what purpose the artifact was used with. Will be shown in UI. |

| Returns |  |
| :--- | :--- |
|  An `Artifact` object. |

### `use_model`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L3447-L3498)

```python
use_model(
    name: str
) -> FilePathStr
```

Download the files logged in a model artifact 'name'.

| Arguments |  |
| :--- | :--- |
|  `name` |  (str) A model artifact name. 'name' must match the name of an existing logged model artifact. May be prefixed with entity/project/. Valid names can be in the following forms: - model_artifact_name:version - model_artifact_name:alias |

#### Examples:

```python
run.use_model(
    name="my_model_artifact:latest",
)

run.use_model(
    name="my_project/my_model_artifact:v0",
)

run.use_model(
    name="my_entity/my_project/my_model_artifact:<digest>",
)
```

Invalid usage

```python
run.use_model(
    name="my_entity/my_project/my_model_artifact",
)
```

| Raises |  |
| :--- | :--- |
|  `AssertionError` |  if model artifact 'name' is of a type that does not contain the substring 'model'. |

| Returns |  |
| :--- | :--- |
|  `path` |  (str) path to downloaded model artifact file(s). |

### `watch`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L2888-L2898)

```python
watch(
    models, criterion=None, log="gradients", log_freq=100, idx=None,
    log_graph=(False)
) -> None
```

### `__enter__`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L3629-L3630)

```python
__enter__() -> "Run"
```

### `__exit__`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L3632-L3643)

```python
__exit__(
    exc_type: Type[BaseException],
    exc_val: BaseException,
    exc_tb: TracebackType
) -> bool
```
