# log

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L1678-L1933' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


Upload run data.

```python
log(
    data: Dict[str, Any],
    step: Optional[int] = None,
    commit: Optional[bool] = None,
    sync: Optional[bool] = None
) -> None
```

Use `log` to log data from runs, such as scalars, images, video,
histograms, plots, and tables.

See our [guides to logging](/guides/track/log) for
live examples, code snippets, best practices, and more.

The most basic usage is `run.log({"train-loss": 0.5, "accuracy": 0.9})`.
This will save the loss and accuracy to the run's history and update
the summary values for these metrics.

Visualize logged data in the workspace at [wandb.ai](https://wandb.ai),
or locally on a [self-hosted instance](/guides/hosting)
of the W&B app, or export data to visualize and explore locally, e.g. in
Jupyter notebooks, with [our API](/guides/track/public-api-guide).

Logged values don't have to be scalars. Logging any wandb object is supported.
For example `run.log({"example": wandb.Image("myimage.jpg")})` will log an
example image which will be displayed nicely in the W&B UI.
See the [reference documentation](/ref/python/data-types)
for all of the different supported types or check out our
[guides to logging](/guides/track/log) for examples,
from 3D molecular structures and segmentation masks to PR curves and histograms.
You can use `wandb.Table` to log structured data. See our
[guide to logging tables](/guides/data-vis/log-tables)
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

See also [define_metric](/ref/python/run#define_metric).

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
[our guides to logging](/guides/track/log).

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
