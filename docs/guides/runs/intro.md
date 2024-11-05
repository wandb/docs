---
slug: /guides/runs
description: Learn about the basic building block of W&B, Runs.
displayed_sidebar: default
title: Runs
---

A single unit of computation logged by W&B is called a *run*. You can think of a W&B run as an atomic element of your whole project. In other words, each run is a record of a specific computation, such as training a model and logging the results, hyperparameter sweeps, and so forth.

Common patterns for initiating a run include, but are not limited to: 

* Training a model
* Changing a hyperparameter and conducting a new experiment
* Conducting a new machine learning experiment with a different model
* Logging data or a model as a [W&B Artifact](../artifacts/intro.md)
* [Downloading a W&B Artifact](../artifacts/download-and-use-an-artifact.md)


W&B stores runs that you create into [*projects*](../track/project-page.md). You can view runs and their properties within the run's project workspace on the W&B App UI. You can also programmatically access run properties with the [`wandb.Run`](../../ref/python/run.md) object.

<!-- Need to verify last sentence. -->

Anything that you log within a run is logged to that run. For example, consider the proceeding code snippet:

```python
import wandb

run = wandb.init(entity="nico", project="awesome-project")

run.log({"accuracy": 0.9, "loss": 0.1})

run.finish()
```

This returns:

```bash
wandb: Syncing run earnest-sunset-1
wandb: ⭐️ View project at https://wandb.ai/nico/awesome-project
wandb: 🚀 View run at https://wandb.ai/nico/awesome-project/runs/1jx1ud12
wandb:                                                                                
wandb: 
wandb: Run history:
wandb: accuracy ▁
wandb:     loss ▁
wandb: 
wandb: Run summary:
wandb: accuracy 0.9
wandb:     loss 0.5
wandb: 
wandb: 🚀 View run earnest-sunset-1 at: https://wandb.ai/nico/awesome-project/runs/1jx1ud12
wandb: ⭐️ View project at: https://wandb.ai/nico/awesome-project
wandb: Synced 6 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20241105_111006-1jx1ud12/logs
```

<!-- Describe what the UI shows (and how to get there) -->


Logging a metrics at a single point of time might not be that useful. A more realistic example in the case of training discriminitive models is to log metrics at regular intervals. For example, consider the proceeding code snippet:

```python
epochs = 10
lr = 0.01

run = wandb.init(
    entity="nico",
    project="awesome-project",
    config={
        "learning_rate": lr,
        "epochs": epochs,
    },
)

offset = random.random() / 5

# simulating a training run
for epoch in range(epochs):
    acc = 1 - 2**-epoch - random.random() / (epoch+1) - offset
    loss = 2**-epoch + random.random() /  (epoch+1) + offset
    print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
    run.log({"accuracy": acc, "loss": loss})
```


This returns the following output:

```bash
wandb: Syncing run jolly-haze-4
wandb: ⭐️ View project at https://wandb.ai/nico/awesome-project
wandb: 🚀 View run at https://wandb.ai/nico/awesome-project/runs/pdo5110r
lr: 0.01
epoch=0, accuracy=-0.10070974957523078, loss=1.985328507123956
epoch=1, accuracy=0.2884687745057535, loss=0.7374362314407752
epoch=2, accuracy=0.7347387967382066, loss=0.4402409835486663
epoch=3, accuracy=0.7667969248039795, loss=0.26176963846423457
epoch=4, accuracy=0.7446848791003173, loss=0.24808611724405083
epoch=5, accuracy=0.8035095836268268, loss=0.16169791827329466
epoch=6, accuracy=0.861349032371624, loss=0.03432578493587426
epoch=7, accuracy=0.8794926436276016, loss=0.10331872172219471
epoch=8, accuracy=0.9424839917077272, loss=0.07767793473500445
epoch=9, accuracy=0.9584880427028566, loss=0.10531971149250456
wandb: 🚀 View run jolly-haze-4 at: https://wandb.ai/nico/awesome-project/runs/pdo5110r
wandb: Find logs at: wandb/run-20241105_111816-pdo5110r/logs
```

In the training script `run.log` is called 10 times. Each time it is called, it logs the accuracy and loss for that epoch. Selecting the URL that is returned from the preceding output, directs you to the run's workspace in the W&B App UI.

Note that the simulated training loop is captured within a single run called `jolly-haze-4`. This is becaeuse the `wandb.init` method is called only once. The `run` object is used to log metrics at each epoch.

![](/images/runs/run_log_example_2.png)


This is not always the case. For example, during a [sweep](../sweeps/intro.md), W&B explores a hyperparameter search space that you specify. Each new hyperparameter combination created by the sweep is implemented and recorded as a unique run. 

<!-- :::info
Some key things to consider when you create and manage runs:
* Anything you log with `wandb.log` is recorded in that run.  For more information on how log objects in W&B, see [Log Media and Objects](../track/log/intro.md). 
* View runs and their properties within the run's project workspace on the W&B App UI.
* There is only at most one active [`wandb.Run`](../../ref/python/run.md) in any process,
and it is accessible as `wandb.run`.
::: -->

