---
title: Run
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/wandb_run.py#L477-L4012 >}}

A unit of computation logged by W&B. Typically, this is an ML experiment.

Call [`wandb.init()`](https://docs.wandb.ai/ref/python/init/) to create a
new run. `wandb.init()` starts a new run and returns a `wandb.Run` object.
Each run is associated with a unique ID (run ID). W&B recommends using
a context (`with` statement) manager to automatically finish the run.

For distributed training experiments, you can either track each process
separately using one run per process or track all processes to a single run.
See [Log distributed training experiments](https://docs.wandb.ai/guides/track/log/distributed-training)
for more information.

You can log data to a run with `wandb.Run.log()`. Anything you log using
`wandb.Run.log()` is sent to that run. See
[Create an experiment](https://docs.wandb.ai/guides/track/launch) or
[`wandb.init`](https://docs.wandb.ai/ref/python/init/) API reference page
or more information.

There is a another `Run` object in the
[`wandb.apis.public`](https://docs.wandb.ai/ref/python/public-api/api/)
namespace. Use this object is to interact with runs that have already been
created.

#### Examples:

Create a run with `wandb.init()`:

```python
import wandb

# Start a new run and log some data
# Use context manager (`with` statement) to automatically finish the run
with wandb.init(entity="entity", project="project") as run:
    run.log({"accuracy": acc, "loss": loss})
```

<!-- lazydoc-ignore-init: internal -->


| Attributes |  |
| :--- | :--- |
|  `summary` |  (Summary) A summary of the run, which is a dictionary-like object. For more information, see [Log summary metrics](https://docs.wandb.ai/guides/track/log/log-summary/). |
|  `config` |  Config object associated with this run. |
|  `config_static` |  Static config object associated with this run. |
|  `dir` |  The directory where files associated with the run are saved. |
|  `disabled` |  True if the run is disabled, False otherwise. |
|  `entity` |  The name of the W&B entity associated with the run. Entity can be a username or the name of a team or organization. |
|  `group` |  Returns the name of the group associated with this run. Grouping runs together allows related experiments to be organized and visualized collectively in the W&B UI. This is especially useful for scenarios such as distributed training or cross-validation, where multiple runs should be viewed and managed as a unified experiment. In shared mode, where all processes share the same run object, setting a group is usually unnecessary, since there is only one run and no grouping is required. |
|  `id` |  Identifier for this run. |
|  `job_type` |  Name of the job type associated with the run. View a run's job type in the run's Overview page in the W&B App. You can use this to categorize runs by their job type, such as "training", "evaluation", or "inference". This is useful for organizing and filtering runs in the W&B UI, especially when you have multiple runs with different job types in the same project. For more information, see [Organize runs](https://docs.wandb.ai/guides/runs/#organize-runs). |
|  `name` |  Display name of the run. Display names are not guaranteed to be unique and may be descriptive. By default, they are randomly generated. |
|  `notes` |  Notes associated with the run, if there are any. Notes can be a multiline string and can also use markdown and latex equations inside `$$`, like `$x + 3$`. |
|  `offline` |  True if the run is offline, False otherwise. |
|  `path` |  Path to the run. Run paths include entity, project, and run ID, in the format `entity/project/run_id`. |
|  `project` |  Name of the W&B project associated with the run. |
|  `project_url` |  URL of the W&B project associated with the run, if there is one. Offline runs do not have a project URL. |
|  `resumed` |  True if the run was resumed, False otherwise. |
|  `settings` |  A frozen copy of run's Settings object. |
|  `start_time` |  Unix timestamp (in seconds) of when the run started. |
|  `starting_step` |  The first step of the run. <!-- lazydoc-ignore: internal --> |
|  `step` |  Current value of the step. This counter is incremented by `wandb.Run.log()`. <!-- lazydoc-ignore: internal --> |
|  `sweep_id` |  Identifier for the sweep associated with the run, if there is one. |
|  `sweep_url` |  URL of the sweep associated with the run, if there is one. Offline runs do not have a sweep URL. |
|  `tags` |  Tags associated with the run, if there are any. |
|  `url` |  The url for the W&B run, if there is one. Offline runs will not have a url. |

## Methods

### `alert`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/wandb_run.py#L3589-L3623)

```python
alert(
    title: str,
    text: str,
    level: (str | AlertLevel | None) = None,
    wait_duration: (int | float | timedelta | None) = None
) -> None
```

Create an alert with the given title and text.

| Args |  |
| :--- | :--- |
|  `title` |  The title of the alert, must be less than 64 characters long. |
|  `text` |  The text body of the alert. |
|  `level` |  The alert level to use, either: `INFO`, `WARN`, or `ERROR`. |
|  `wait_duration` |  The time to wait (in seconds) before sending another alert with this title. |

### `define_metric`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/wandb_run.py#L2780-L2845)

```python
define_metric(
    name: str,
    step_metric: (str | wandb_metric.Metric | None) = None,
    step_sync: (bool | None) = None,
    hidden: (bool | None) = None,
    summary: (str | None) = None,
    goal: (str | None) = None,
    overwrite: (bool | None) = None
) -> wandb_metric.Metric
```

Customize metrics logged with `wandb.Run.log()`.

| Args |  |
| :--- | :--- |
|  `name` |  The name of the metric to customize. |
|  `step_metric` |  The name of another metric to serve as the X-axis for this metric in automatically generated charts. |
|  `step_sync` |  Automatically insert the last value of step_metric into `wandb.Run.log()` if it is not provided explicitly. Defaults to True if step_metric is specified. |
|  `hidden` |  Hide this metric from automatic plots. |
|  `summary` |  Specify aggregate metrics added to summary. Supported aggregations include "min", "max", "mean", "last", "first", "best", "copy" and "none". "none" prevents a summary from being generated. "best" is used together with the goal parameter, "best" is deprecated and should not be used, use "min" or "max" instead. "copy" is deprecated and should not be used. |
|  `goal` |  Specify how to interpret the "best" summary type. Supported options are "minimize" and "maximize". "goal" is deprecated and should not be used, use "min" or "max" instead. |
|  `overwrite` |  If false, then this call is merged with previous `define_metric` calls for the same metric by using their values for any unspecified parameters. If true, then unspecified parameters overwrite values specified by previous calls. |

| Returns |  |
| :--- | :--- |
|  An object that represents this call but can otherwise be discarded. |

### `display`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/wandb_run.py#L1345-L1362)

```python
display(
    height: int = 420,
    hidden: bool = (False)
) -> bool
```

Display this run in Jupyter.

### `finish`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/wandb_run.py#L2227-L2260)

```python
finish(
    exit_code: (int | None) = None,
    quiet: (bool | None) = None
) -> None
```

Finish a run and upload any remaining data.

Marks the completion of a W&B run and ensures all data is synced to the server.
The run's final state is determined by its exit conditions and sync status.

#### Run States:

- Running: Active run that is logging data and/or sending heartbeats.
- Crashed: Run that stopped sending heartbeats unexpectedly.
- Finished: Run completed successfully (`exit_code=0`) with all data synced.
- Failed: Run completed with errors (`exit_code!=0`).
- Killed: Run was forcibly stopped before it could finish.

| Args |  |
| :--- | :--- |
|  `exit_code` |  Integer indicating the run's exit status. Use 0 for success, any other value marks the run as failed. |
|  `quiet` |  Deprecated. Configure logging verbosity using `wandb.Settings(quiet=...)`. |

### `finish_artifact`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/wandb_run.py#L3229-L3282)

```python
finish_artifact(
    artifact_or_path: (Artifact | str),
    name: (str | None) = None,
    type: (str | None) = None,
    aliases: (list[str] | None) = None,
    distributed_id: (str | None) = None
) -> Artifact
```

Finishes a non-finalized artifact as output of a run.

Subsequent "upserts" with the same distributed ID will result in a new version.

| Args |  |
| :--- | :--- |
|  `artifact_or_path` |  A path to the contents of this artifact, can be in the following forms: - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` You can also pass an Artifact object created by calling `wandb.Artifact`. |
|  `name` |  An artifact name. May be prefixed with entity/project. Valid names can be in the following forms: - name:version - name:alias - digest This will default to the basename of the path prepended with the current run id if not specified. |
|  `type` |  The type of artifact to log, examples include `dataset`, `model` |
|  `aliases` |  Aliases to apply to this artifact, defaults to `["latest"]` |
|  `distributed_id` |  Unique string that all distributed jobs share. If None, defaults to the run's group name. |

| Returns |  |
| :--- | :--- |
|  An `Artifact` object. |

### `get_project_url`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/wandb_run.py#L1055-L1071)

```python
get_project_url() -> (str | None)
```

This method is deprecated and will be removed in a future release. Use `run.project_url` instead.

URL of the W&B project associated with the run, if there is one.
Offline runs do not have a project URL.

<!-- lazydoc-ignore: internal -->


### `get_sweep_url`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/wandb_run.py#L1189-L1205)

```python
get_sweep_url() -> (str | None)
```

This method is deprecated and will be removed in a future release. Use `run.sweep_url` instead.

The URL of the sweep associated with the run, if there is one.
Offline runs do not have a sweep URL.

<!-- lazydoc-ignore: internal -->


### `get_url`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/wandb_run.py#L1219-L1234)

```python
get_url() -> (str | None)
```

This method is deprecated and will be removed in a future release. Use `run.url` instead.

URL of the W&B run, if there is one. Offline runs do not have a URL.

<!-- lazydoc-ignore: internal -->


### `link_artifact`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/wandb_run.py#L2968-L3010)

```python
link_artifact(
    artifact: Artifact,
    target_path: str,
    aliases: (list[str] | None) = None
) -> Artifact
```

Link the given artifact to a portfolio (a promoted collection of artifacts).

Linked artifacts are visible in the UI for the specified portfolio.

| Args |  |
| :--- | :--- |
|  `artifact` |  the (public or local) artifact which will be linked |
|  `target_path` |  `str` - takes the following forms: `{portfolio}`, `{project}/{portfolio}`, or `{entity}/{project}/{portfolio}` |
|  `aliases` |  `List[str]` - optional alias(es) that will only be applied on this linked artifact inside the portfolio. The alias "latest" will always be applied to the latest version of an artifact that is linked. |

| Returns |  |
| :--- | :--- |
|  The linked artifact. |

### `link_model`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/wandb_run.py#L3508-L3587)

```python
link_model(
    path: StrPath,
    registered_model_name: str,
    name: (str | None) = None,
    aliases: (list[str] | None) = None
) -> (Artifact | None)
```

Log a model artifact version and link it to a registered model in the model registry.

Linked model versions are visible in the UI for the specified registered model.

#### This method will:

- Check if 'name' model artifact has been logged. If so, use the artifact version that matches the files
  located at 'path' or log a new version. Otherwise log files under 'path' as a new model artifact, 'name'
  of type 'model'.
- Check if registered model with name 'registered_model_name' exists in the 'model-registry' project.
  If not, create a new registered model with name 'registered_model_name'.
- Link version of model artifact 'name' to registered model, 'registered_model_name'.
- Attach aliases from 'aliases' list to the newly linked model artifact version.

| Args |  |
| :--- | :--- |
|  `path` |  (str) A path to the contents of this model, can be in the following forms: - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` |
|  `registered_model_name` |  The name of the registered model that the model is to be linked to. A registered model is a collection of model versions linked to the model registry, typically representing a team's specific ML Task. The entity that this registered model belongs to will be derived from the run. |
|  `name` |  The name of the model artifact that files in 'path' will be logged to. This will default to the basename of the path prepended with the current run id if not specified. |
|  `aliases` |  Aliases that will only be applied on this linked artifact inside the registered model. The alias "latest" will always be applied to the latest version of an artifact that is linked. |

| Raises |  |
| :--- | :--- |
|  `AssertionError` |  If registered_model_name is a path or if model artifact 'name' is of a type that does not contain the substring 'model'. |
|  `ValueError` |  If name has invalid special characters. |

| Returns |  |
| :--- | :--- |
|  The linked artifact if linking was successful, otherwise `None`. |

### `log`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/wandb_run.py#L1765-L2028)

```python
log(
    data: dict[str, Any],
    step: (int | None) = None,
    commit: (bool | None) = None
) -> None
```

Upload run data.

Use `log` to log data from runs, such as scalars, images, video,
histograms, plots, and tables. See [Log objects and media](https://docs.wandb.ai/guides/track/log) for
code snippets, best practices, and more.

#### Basic usage:

```python
import wandb

with wandb.init() as run:
    run.log({"train-loss": 0.5, "accuracy": 0.9})
```

The previous code snippet saves the loss and accuracy to the run's
history and updates the summary values for these metrics.

Visualize logged data in a workspace at [wandb.ai](https://wandb.ai),
or locally on a [self-hosted instance](https://docs.wandb.ai/guides/hosting)
of the W&B app, or export data to visualize and explore locally, such as in a
Jupyter notebook, with the [Public API](https://docs.wandb.ai/guides/track/public-api-guide).

Logged values don't have to be scalars. You can log any
[W&B supported Data Type](https://docs.wandb.ai/ref/python/data-types/)
such as images, audio, video, and more. For example, you can use
`wandb.Table` to log structured data. See
[Log tables, visualize and query data](https://docs.wandb.ai/guides/models/tables/tables-walkthrough)
tutorial for more details.

W&B organizes metrics with a forward slash (`/`) in their name
into sections named using the text before the final slash. For example,
the following results in two sections named "train" and "validate":

```python
with wandb.init() as run:
    # Log metrics in the "train" section.
    run.log(
        {
            "train/accuracy": 0.9,
            "train/loss": 30,
            "validate/accuracy": 0.8,
            "validate/loss": 20,
        }
    )
```

Only one level of nesting is supported; `run.log({"a/b/c": 1})`
produces a section named "a/b".

`run.log()` is not intended to be called more than a few times per second.
For optimal performance, limit your logging to once every N iterations,
or collect data over multiple iterations and log it in a single step.

By default, each call to `log` creates a new "step".
The step must always increase, and it is not possible to log
to a previous step. You can use any metric as the X axis in charts.
See [Custom log axes](https://docs.wandb.ai/guides/track/log/customize-logging-axes/)
for more details.

In many cases, it is better to treat the W&B step like
you'd treat a timestamp rather than a training step.

```python
with wandb.init() as run:
    # Example: log an "epoch" metric for use as an X axis.
    run.log({"epoch": 40, "train-loss": 0.5})
```

It is possible to use multiple `wandb.Run.log()` invocations to log to
the same step with the `step` and `commit` parameters.
The following are all equivalent:

```python
with wandb.init() as run:
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

| Args |  |
| :--- | :--- |
|  `data` |  A `dict` with `str` keys and values that are serializable Python objects including: `int`, `float` and `string`; any of the `wandb.data_types`; lists, tuples and NumPy arrays of serializable Python objects; other `dict`s of this structure. |
|  `step` |  The step number to log. If `None`, then an implicit auto-incrementing step is used. See the notes in the description. |
|  `commit` |  If true, finalize and upload the step. If false, then accumulate data for the step. See the notes in the description. If `step` is `None`, then the default is `commit=True`; otherwise, the default is `commit=False`. |

#### Examples:

For more and more detailed examples, see
[our guides to logging](https://docs.wandb.com/guides/track/log).

Basic usage

```python
import wandb

with wandb.init() as run:
    run.log({"train-loss": 0.5, "accuracy": 0.9
```

Incremental logging

```python
import wandb

with wandb.init() as run:
    run.log({"loss": 0.2}, commit=False)
    # Somewhere else when I'm ready to report this step:
    run.log({"accuracy": 0.8})
```

Histogram

```python
import numpy as np
import wandb

# sample gradients at random from normal distribution
gradients = np.random.randn(100, 100)
with wandb.init() as run:
    run.log({"gradients": wandb.Histogram(gradients)})
```

Image from NumPy

```python
import numpy as np
import wandb

with wandb.init() as run:
    examples = []
    for i in range(3):
        pixels = np.random.randint(low=0, high=256, size=(100, 100, 3))
        image = wandb.Image(pixels, caption=f"random field {i}")
        examples.append(image)
    run.log({"examples": examples})
```

Image from PIL

```python
import numpy as np
from PIL import Image as PILImage
import wandb

with wandb.init() as run:
    examples = []
    for i in range(3):
        pixels = np.random.randint(
            low=0,
            high=256,
            size=(100, 100, 3),
            dtype=np.uint8,
        )
        pil_image = PILImage.fromarray(pixels, mode="RGB")
        image = wandb.Image(pil_image, caption=f"random field {i}")
        examples.append(image)
    run.log({"examples": examples})
```

Video from NumPy

```python
import numpy as np
import wandb

with wandb.init() as run:
    # axes are (time, channel, height, width)
    frames = np.random.randint(
        low=0,
        high=256,
        size=(10, 3, 100, 100),
        dtype=np.uint8,
    )
    run.log({"video": wandb.Video(frames, fps=4)})
```

Matplotlib plot

```python
from matplotlib import pyplot as plt
import numpy as np
import wandb

with wandb.init() as run:
    fig, ax = plt.subplots()
    x = np.linspace(0, 10)
    y = x * x
    ax.plot(x, y)  # plot y = x^2
    run.log({"chart": fig})
```

PR Curve

```python
import wandb

with wandb.init() as run:
    run.log({"pr": wandb.plot.pr_curve(y_test, y_probas, labels)})
```

3D Object

```python
import wandb

with wandb.init() as run:
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
|  `wandb.Error` |  If called before `wandb.init()`. |
|  `ValueError` |  If invalid data is passed. |

### `log_artifact`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/wandb_run.py#L3135-L3176)

```python
log_artifact(
    artifact_or_path: (Artifact | StrPath),
    name: (str | None) = None,
    type: (str | None) = None,
    aliases: (list[str] | None) = None,
    tags: (list[str] | None) = None
) -> Artifact
```

Declare an artifact as an output of a run.

| Args |  |
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

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/wandb_run.py#L1086-L1187)

```python
log_code(
    root: (str | None) = ".",
    name: (str | None) = None,
    include_fn: (Callable[[str, str], bool] | Callable[[str], bool]) = _is_py_requirements_or_dockerfile,
    exclude_fn: (Callable[[str, str], bool] | Callable[[str], bool]) = filenames.exclude_wandb_fn
) -> (Artifact | None)
```

Save the current state of your code to a W&B Artifact.

By default, it walks the current directory and logs all files that end with `.py`.

| Args |  |
| :--- | :--- |
|  `root` |  The relative (to `os.getcwd()`) or absolute path to recursively find code from. |
|  `name` |  (str, optional) The name of our code artifact. By default, we'll name the artifact `source-$PROJECT_ID-$ENTRYPOINT_RELPATH`. There may be scenarios where you want many runs to share the same artifact. Specifying name allows you to achieve that. |
|  `include_fn` |  A callable that accepts a file path and (optionally) root path and returns True when it should be included and False otherwise. This defaults to `lambda path, root: path.endswith(".py")`. |
|  `exclude_fn` |  A callable that accepts a file path and (optionally) root path and returns `True` when it should be excluded and `False` otherwise. This defaults to a function that excludes all files within `<root>/.wandb/` and `<root>/wandb/` directories. |

#### Examples:

Basic usage

```python
import wandb

with wandb.init() as run:
    run.log_code()
```

Advanced usage

```python
import wandb

with wandb.init() as run:
    run.log_code(
        root="../",
        include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"),
        exclude_fn=lambda path, root: os.path.relpath(path, root).startswith(
            "cache/"
        ),
    )
```

| Returns |  |
| :--- | :--- |
|  An `Artifact` object if code was logged |

### `log_model`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/wandb_run.py#L3431-L3466)

```python
log_model(
    path: StrPath,
    name: (str | None) = None,
    aliases: (list[str] | None) = None
) -> None
```

Logs a model artifact containing the contents inside the 'path' to a run and marks it as an output to this run.

The name of model artifact can only contain alphanumeric characters,
underscores, and hyphens.

| Args |  |
| :--- | :--- |
|  `path` |  (str) A path to the contents of this model, can be in the following forms: - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` |
|  `name` |  A name to assign to the model artifact that the file contents will be added to. This will default to the basename of the path prepended with the current run id if not specified. |
|  `aliases` |  Aliases to apply to the created model artifact, defaults to `["latest"]` |

| Raises |  |
| :--- | :--- |
|  `ValueError` |  If name has invalid special characters. |

| Returns |  |
| :--- | :--- |
|  None |

### `mark_preempting`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/wandb_run.py#L3641-L3650)

```python
mark_preempting() -> None
```

Mark this run as preempting.

Also tells the internal process to immediately report this to server.

### `project_name`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/wandb_run.py#L1031-L1045)

```python
project_name() -> str
```

This method is deprecated and will be removed in a future release. Use `run.project` instead.

Name of the W&B project associated with the run.

<!-- lazydoc-ignore: internal -->


### `restore`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/wandb_run.py#L2211-L2225)

```python
restore(
    name: str,
    run_path: (str | None) = None,
    replace: bool = (False),
    root: (str | None) = None
) -> (None | TextIO)
```

Download the specified file from cloud storage.

File is placed into the current directory or run directory.
By default, will only download the file if it doesn't already exist.

| Args |  |
| :--- | :--- |
|  `name` |  The name of the file. |
|  `run_path` |  Optional path to a run to pull files from, i.e. `username/project_name/run_id` if wandb.init has not been called, this is required. |
|  `replace` |  Whether to download the file even if it already exists locally |
|  `root` |  The directory to download the file to. Defaults to the current directory or the run directory if wandb.init was called. |

| Returns |  |
| :--- | :--- |
|  None if it can't find the file, otherwise a file object open for reading. |

| Raises |  |
| :--- | :--- |
|  `CommError` |  If W&B can't connect to the W&B backend. |
|  `ValueError` |  If the file is not found or can't find run_path. |

### `save`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/wandb_run.py#L2030-L2130)

```python
save(
    glob_str: (str | os.PathLike),
    base_path: (str | os.PathLike | None) = None,
    policy: PolicyName = "live"
) -> (bool | list[str])
```

Sync one or more files to W&B.

Relative paths are relative to the current working directory.

A Unix glob, such as "myfiles/*", is expanded at the time `save` is
called regardless of the `policy`. In particular, new files are not
picked up automatically.

A `base_path` may be provided to control the directory structure of
uploaded files. It should be a prefix of `glob_str`, and the directory
structure beneath it is preserved.

When given an absolute path or glob and no `base_path`, one
directory level is preserved as in the example above.

| Args |  |
| :--- | :--- |
|  `glob_str` |  A relative or absolute path or Unix glob. |
|  `base_path` |  A path to use to infer a directory structure; see examples. |
|  `policy` |  One of `live`, `now`, or `end`. - live: upload the file as it changes, overwriting the previous version - now: upload the file once now - end: upload file when the run ends |

| Returns |  |
| :--- | :--- |
|  Paths to the symlinks created for the matched files. For historical reasons, this may return a boolean in legacy code. |

```python
import wandb

run = wandb.init()

run.save("these/are/myfiles/*")
# => Saves files in a "these/are/myfiles/" folder in the run.

run.save("these/are/myfiles/*", base_path="these")
# => Saves files in an "are/myfiles/" folder in the run.

run.save("/User/username/Documents/run123/*.txt")
# => Saves files in a "run123/" folder in the run. See note below.

run.save("/User/username/Documents/run123/*.txt", base_path="/User")
# => Saves files in a "username/Documents/run123/" folder in the run.

run.save("files/*/saveme.txt")
# => Saves each "saveme.txt" file in an appropriate subdirectory
#    of "files/".

# Explicitly finish the run since a context manager is not used.
run.finish()
```

### `status`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/wandb_run.py#L2309-L2332)

```python
status() -> RunStatus
```

Get sync info from the internal backend, about the current run's sync status.

### `to_html`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/wandb_run.py#L1364-L1377)

```python
to_html(
    height: int = 420,
    hidden: bool = (False)
) -> str
```

Generate HTML containing an iframe displaying the current run.

<!-- lazydoc-ignore: internal -->


### `unwatch`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/wandb_run.py#L2956-L2966)

```python
unwatch(
    models: (torch.nn.Module | Sequence[torch.nn.Module] | None) = None
) -> None
```

Remove pytorch model topology, gradient and parameter hooks.

| Args |  |
| :--- | :--- |
|  `models` |  Optional list of pytorch models that have had watch called on them. |

### `upsert_artifact`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/wandb_run.py#L3178-L3227)

```python
upsert_artifact(
    artifact_or_path: (Artifact | str),
    name: (str | None) = None,
    type: (str | None) = None,
    aliases: (list[str] | None) = None,
    distributed_id: (str | None) = None
) -> Artifact
```

Declare (or append to) a non-finalized artifact as output of a run.

Note that you must call run.finish_artifact() to finalize the artifact.
This is useful when distributed jobs need to all contribute to the same artifact.

| Args |  |
| :--- | :--- |
|  `artifact_or_path` |  A path to the contents of this artifact, can be in the following forms: - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` |
|  `name` |  An artifact name. May be prefixed with "entity/project". Defaults to the basename of the path prepended with the current run ID if not specified. Valid names can be in the following forms: - name:version - name:alias - digest |
|  `type` |  The type of artifact to log. Common examples include `dataset`, `model`. |
|  `aliases` |  Aliases to apply to this artifact, defaults to `["latest"]`. |
|  `distributed_id` |  Unique string that all distributed jobs share. If None, defaults to the run's group name. |

| Returns |  |
| :--- | :--- |
|  An `Artifact` object. |

### `use_artifact`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/wandb_run.py#L3012-L3133)

```python
use_artifact(
    artifact_or_name: (str | Artifact),
    type: (str | None) = None,
    aliases: (list[str] | None) = None,
    use_as: (str | None) = None
) -> Artifact
```

Declare an artifact as an input to a run.

Call `download` or `file` on the returned object to get the contents locally.

| Args |  |
| :--- | :--- |
|  `artifact_or_name` |  The name of the artifact to use. May be prefixed with the name of the project the artifact was logged to ("<entity>" or "<entity>/<project>"). If no entity is specified in the name, the Run or API setting's entity is used. Valid names can be in the following forms - name:version - name:alias |
|  `type` |  The type of artifact to use. |
|  `aliases` |  Aliases to apply to this artifact |
|  `use_as` |  This argument is deprecated and does nothing. |

| Returns |  |
| :--- | :--- |
|  An `Artifact` object. |

#### Examples:

```python
import wandb

run = wandb.init(project="<example>")

# Use an artifact by name and alias
artifact_a = run.use_artifact(artifact_or_name="<name>:<alias>")

# Use an artifact by name and version
artifact_b = run.use_artifact(artifact_or_name="<name>:v<version>")

# Use an artifact by entity/project/name:alias
artifact_c = run.use_artifact(
    artifact_or_name="<entity>/<project>/<name>:<alias>"
)

# Use an artifact by entity/project/name:version
artifact_d = run.use_artifact(
    artifact_or_name="<entity>/<project>/<name>:v<version>"
)

# Explicitly finish the run since a context manager is not used.
run.finish()
```

### `use_model`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/wandb_run.py#L3468-L3506)

```python
use_model(
    name: str
) -> FilePathStr
```

Download the files logged in a model artifact 'name'.

| Args |  |
| :--- | :--- |
|  `name` |  A model artifact name. 'name' must match the name of an existing logged model artifact. May be prefixed with `entity/project/`. Valid names can be in the following forms - model_artifact_name:version - model_artifact_name:alias |

| Returns |  |
| :--- | :--- |
|  path (str): Path to downloaded model artifact file(s). |

| Raises |  |
| :--- | :--- |
|  `AssertionError` |  If model artifact 'name' is of a type that does not contain the substring 'model'. |

### `watch`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/wandb_run.py#L2925-L2954)

```python
watch(
    models: (torch.nn.Module | Sequence[torch.nn.Module]),
    criterion: (torch.F | None) = None,
    log: (Literal['gradients', 'parameters', 'all'] | None) = "gradients",
    log_freq: int = 1000,
    idx: (int | None) = None,
    log_graph: bool = (False)
) -> None
```

Hook into given PyTorch model to monitor gradients and the model's computational graph.

This function can track parameters, gradients, or both during training.

| Args |  |
| :--- | :--- |
|  `models` |  A single model or a sequence of models to be monitored. |
|  `criterion` |  The loss function being optimized (optional). |
|  `log` |  Specifies whether to log "gradients", "parameters", or "all". Set to None to disable logging. (default="gradients"). |
|  `log_freq` |  Frequency (in batches) to log gradients and parameters. (default=1000) |
|  `idx` |  Index used when tracking multiple models with `wandb.watch`. (default=None) |
|  `log_graph` |  Whether to log the model's computational graph. (default=False) |

| Raises |  |
| :--- | :--- |
|  `ValueError` |  If `wandb.init()` has not been called or if any of the models are not instances of `torch.nn.Module`. |

### `__enter__`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/wandb_run.py#L3625-L3626)

```python
__enter__() -> Run
```

### `__exit__`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/sdk/wandb_run.py#L3628-L3639)

```python
__exit__(
    exc_type: type[BaseException],
    exc_val: BaseException,
    exc_tb: TracebackType
) -> bool
```
