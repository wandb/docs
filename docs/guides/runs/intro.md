---
slug: /guides/runs
description: Learn about the basic building block of W&B, Runs.
displayed_sidebar: default
title: Runs
---

A single unit of computation logged by W&B is called a *run*. You can think of a W&B run as an atomic element of your whole project. You should initiate a new run when you:

* Train a model
* Change a hyperparameter
* Use a different model
* Log data or a model as a [W&B Artifact](../artifacts/intro.md)
* [Download a W&B Artifact](../artifacts/download-and-use-an-artifact.md)

For example, during a [sweep](../sweeps/intro.md), W&B explores a hyperparameter search space that you specify. Each new hyperparameter combination created by the sweep is implemented and recorded as a unique run. 


:::tip
Some key things to consider when you create and manage runs:
* Anything you log with `wandb.log` is recorded in that run.  For more information on how log objects in W&B, see [Log Media and Objects](../track/log/intro.md). 
* Each run is associated to a specific W&B project.
* View runs and their properties within the run's project workspace on the W&B App UI.
* There is only at most one active [`wandb.Run`](../../ref/python/run.md) in any process,
and it is accessible as `wandb.run`.
:::

