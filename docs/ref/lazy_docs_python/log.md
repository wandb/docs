import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

# log

<CTAButtons githubLink='https://github.com/wandb/wandb/blob/main/wandb/sdk/lib/preinit.py'/>




### <kbd>function</kbd> `wandb.log`

```python
wandb.log(
    data: 'dict[str, Any]',
    step: 'int | None' = None,
    commit: 'bool | None' = None,
    sync: 'bool | None' = None
) → None
```

Upload run data. 

Use `log` to log data from runs, such as scalars, images, video, histograms, plots, and tables. 

See our [guides to logging](https://docs.wandb.ai/guides/track/log) for live examples, code snippets, best practices, and more. 

The most basic usage is `run.log({"train-loss": 0.5, "accuracy": 0.9})`. This will save the loss and accuracy to the run's history and update the summary values for these metrics. 

Visualize logged data in the workspace at [wandb.ai](https://wandb.ai), or locally on a [self-hosted instance](https://docs.wandb.ai/guides/hosting) of the W&B app, or export data to visualize and explore locally, e.g. in Jupyter notebooks, with [our API](https://docs.wandb.ai/guides/track/public-api-guide). 

Logged values don't have to be scalars. Logging any wandb object is supported. For example `run.log({"example": wandb.Image("myimage.jpg")})` will log an example image which will be displayed nicely in the W&B UI. See the [reference documentation](https://docs.wandb.com/ref/python/data-types) for all of the different supported types or check out our [guides to logging](https://docs.wandb.ai/guides/track/log) for examples, from 3D molecular structures and segmentation masks to PR curves and histograms. You can use `wandb.Table` to log structured data. See our [guide to logging tables](https://docs.wandb.ai/guides/tables/tables-walkthrough) for details. 

The W&B UI organizes metrics with a forward slash (`/`) in their name into sections named using the text before the final slash. For example, the following results in two sections named "train" and "validate": 

```
run.log({
     "train/accuracy": 0.9,
     "train/loss": 30,
     "validate/accuracy": 0.8,
     "validate/loss": 20,
})
``` 

Only one level of nesting is supported; `run.log({"a/b/c": 1})` produces a section named "a/b". 

`run.log` is not intended to be called more than a few times per second. For optimal performance, limit your logging to once every N iterations, or collect data over multiple iterations and log it in a single step. 

### The W&B step 

With basic usage, each call to `log` creates a new "step". The step must always increase, and it is not possible to log to a previous step. 

Note that you can use any metric as the X axis in charts. In many cases, it is better to treat the W&B step like you'd treat a timestamp rather than a training step. 

```
# Example: log an "epoch" metric for use as an X axis.
run.log({"epoch": 40, "train-loss": 0.5})
``` 

See also [define_metric](https://docs.wandb.ai/ref/python/run#define_metric). 

It is possible to use multiple `log` invocations to log to the same step with the `step` and `commit` parameters. The following are all equivalent: 

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



**Args:**
 
 - `data`:  A `dict` with `str` keys and values that are serializable 
 - `Python objects including`:  `int`, `float` and `string`; any of the `wandb.data_types`; lists, tuples and NumPy arrays of serializable Python objects; other `dict`s of this structure. 
 - `step`:  The step number to log. If `None`, then an implicit  auto-incrementing step is used. See the notes in  the description. 
 - `commit`:  If true, finalize and upload the step. If false, then  accumulate data for the step. See the notes in the description.  If `step` is `None`, then the default is `commit=True`;  otherwise, the default is `commit=False`. 
 - `sync`:  This argument is deprecated and does nothing. 




For more and more detailed examples, see [our guides to logging](https://docs.wandb.com/guides/track/log). 


**Raises:**
 
 - `wandb.Error`:  if called before `wandb.init` 
 - `ValueError`:  if invalid data is passed