---
title: log
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.21.1/wandb/sdk/wandb_run.py#L1753-L2016 >}}

Upload run data.

```python
log(
    data: dict[str, Any],
    step: (int | None) = None,
    commit: (bool | None) = None
) -> None
```

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
