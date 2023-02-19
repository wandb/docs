---
slug: /guides/runs
description:  Learn about the basic building block of W&B, Runs.
---
# Runs

A single unit of computation logged by W&B is called a *Run*. 

Consider Runs as an atomic element of your whole project. You should create and initiate a new Run if you change a hyperparameter, use a different model, create a [W&B Artifact](../artifacts/intro.md) and so on.

For example, in a [W&B Sweep](../sweeps/intro.md), W&B explores a hyperparameter search and explores the space of possible models. Each new hyperparameter combination used by W&B Sweeps is implemented as a W&B Run. 

Use W&B Runs for tasks such as:

* Each time you train a model.
* Log data or a model as a [W&B Artifact](../artifacts/intro.md).
* [Download a W&B Artifact](../artifacts/download-and-use-an-artifact.md).


Anything you log with `wandb.log` is recorded in that Run.  For more information on how log objects in W&B, see [Log Media and Objects](../track/log/intro.md).

View Runs associated to a [Project within the Runs Page](#view-runs) in your project. 

## Create a Run

Create a W&B Run with [`wandb.init()`](../../ref/python/init.md):

```python
import wandb

run = wandb.init(project='my-project-name')
```

Optionally provide the name of a project for the `project` field. W&B creates a new project if a project does not already exist with the name you provide. We recommend you specify a project name when you create a Run object. Projects help organize experiments, runs, artifacts, and more in one convenient location called a *Project Workspace*. 

A Project's Workspace gives you a personal sandbox to compare experiments. Use projects to organize models that can be compared, working on the same problem with different architectures, hyperparameters, datasets, preprocessing and so on.


If a project is not specified, the W&B Run is stored in a project called "Uncategorized".




## End a Run
W&B automatically calls [`wandb.finish`](../../ref/python/finish.md) to finalize and cleanup a run. However, if you call [`wandb.init`](../../ref/python/init.md) from a child process, you must explicitly call `wandb.finish` at the end of the child process.


<!-- [INSERT example code] -->



## Manage multiple Runs
There is only ever at most one active [`wandb.Run`](../../ref/python/run.md) in any process,
and it is accessible as `wandb.run`:

```python
import wandb

assert wandb.run is None

wandb.init()

assert wandb.run is not None
```


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


## View Runs
