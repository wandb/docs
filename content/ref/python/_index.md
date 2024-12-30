---
title: Python Library
---
<!-- Insert buttons and diff -->


{{< cta-button githubLink="https://www.github.com/wandb/wandb/tree/v0.18.7/wandb/__init__.py" >}}

Use wandb to track machine learning work.

Train and fine-tune models, manage models from experimentation to production.

For guides and examples, see https://docs.wandb.ai.

For scripts and interactive notebooks, see https://github.com/wandb/examples.

For reference documentation, see https://docs.wandb.com/ref/python.

## Classes

[`class Artifact`](./artifact.md): Flexible and lightweight building block for dataset and model versioning.

[`class Run`](./run.md): A unit of computation logged by wandb. Typically, this is an ML experiment.

## Functions

[`agent(...)`](./agent.md): Start one or more sweep agents.

[`controller(...)`](./controller.md): Public sweep controller constructor.

[`finish(...)`](./finish.md): Finish a run and upload any remaining data.

[`init(...)`](./init.md): Start a new run to track and log to W&B.

[`log(...)`](./log.md): Upload run data.

[`login(...)`](./login.md): Set up W&B login credentials.

[`save(...)`](./save.md): Sync one or more files to W&B.

[`sweep(...)`](./sweep.md): Initialize a hyperparameter sweep.

[`watch(...)`](./watch.md): Hooks into the given PyTorch models to monitor gradients and the model's computational graph.

| Other Members |  |
| :--- | :--- |
|  `__version__`<a id="__version__"></a> |  `'0.19.1'` |
|  `config`<a id="config"></a> |   |
|  `summary`<a id="summary"></a> |   |
