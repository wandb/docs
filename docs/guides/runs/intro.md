# Runs

A single unit of computation logged by W&B is called a Run. In W&B it is common to associate a Run with:

* Each time you train a model.
* Log data or a model as a [W&B Artifact](../artifacts/intro.md).
* [Download a W&B Artifact](../artifacts/download-and-use-an-artifact.md).

Anything you log with `wandb.log` is sent to that Run.

## Create a Run

Create a W&B Run with [`wandb.init()`](../../ref/python/init.md):

```python
import wandb

run = wandb.init(project='my-project-name')
```

There is only ever at most one active [`wandb.Run`](../../ref/python/run.md) in any process,
and it is accessible as `wandb.run`:

```python
import wandb

assert wandb.run is None

wandb.init()

assert wandb.run is not None
```

## End a Run
W&B automatically call [`wandb.finish`](../../ref/python/finish.md) to finalize and cleanup a run. However, if you call [`wandb.init`](../../ref/python/init.md) from a child process, you must explicitly call wandb.finish at the end of the child process.


<!-- [INSERT example code] -->



## Manage multiple Runs
You need to finish a Run that had not completed in order to start one or more Runs in the same notebook or script. End a Run with the [`wandb.finish`](../../ref/python/finish.md) API or end a Run using a `with` statement. 

The following code example demonstrates how to end a Run from a `with` Python statement:

```python
import wandb

wandb.init()
wandb.finish()

assert wandb.run is None

with wandb.init() as run:
 pass # log data here

assert wandb.run is None
```

:::info
The wandb.finish API is automatically called this method when your script exits.
:::


