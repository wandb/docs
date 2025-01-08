---
description: How to integrate W&B with Kubeflow Pipelines.
menu:
  default:
    identifier: kubeflow-pipelines-kfp
    parent: integrations
title: Kubeflow Pipelines (kfp)
weight: 170
---
## Overview

[Kubeflow Pipelines (kfp) ](https://www.kubeflow.org/docs/components/pipelines/overview/)is a platform for building and deploying portable, scalable machine learning (ML) workflows based on Docker containers.

This integration lets users apply decorators to kfp python functional components to automatically log parameters and artifacts to W&B.

This feature was enabled in `wandb==0.12.11` and requires `kfp<2.0.0`

## Quickstart

### Install W&B and login

{{< tabpane text=true >}}

{{% tab header="Notebook" value="notebook" %}}

```bash
!pip install kfp wandb

import wandb
wandb.login()
```

{{% /tab %}}

{{% tab header="Command Line" value="cli" %}}

```python
pip install kfp wandb
wandb login
```

{{% /tab %}}

{{< /tabpane >}}


### Decorate your components

Add the `@wandb_log` decorator and create your components as usual. This will automatically log the input/outputs parameters and artifacts to W&B each time you run your pipeline.

```python
from kfp import components
from wandb.integration.kfp import wandb_log


@wandb_log
def add(a: float, b: float) -> float:
    return a + b


add = components.create_component_from_func(add)
```

### Pass environment variables to containers

You may need to explicitly pass [environment variables](../../track/environment-variables.md) to your containers. For two-way linking, you should also set the environment variables `WANDB_KUBEFLOW_URL` to the base URL of your Kubeflow Pipelines instance. For example, `https://kubeflow.mysite.com`.

```python
import os
from kubernetes.client.models import V1EnvVar


def add_wandb_env_variables(op):
    env = {
        "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
        "WANDB_BASE_URL": os.getenv("WANDB_BASE_URL"),
    }

    for name, value in env.items():
        op = op.add_env_variable(V1EnvVar(name, value))
    return op


@dsl.pipeline(name="example-pipeline")
def example_pipeline(param1: str, param2: int):
    conf = dsl.get_pipeline_conf()
    conf.add_op_transformer(add_wandb_env_variables)
```

## Access your data programmatically

### Via the Kubeflow Pipelines UI

Click on any Run in the Kubeflow Pipelines UI that has been logged with W&B.

* Find details about inputs and outputs in the `Input/Output` and `ML Metadata` tabs.
* View the W&B web app from the `Visualizations` tab.

{{< img src="/images/integrations/kubeflow_app_pipelines_ui.png" alt="Get a view of W&B in the Kubeflow UI" >}}

### Via the web app UI

The web app UI has the same content as the `Visualizations` tab in Kubeflow Pipelines, but with more space. Learn [more about the web app UI here](/guides/app).

{{< img src="/images/integrations/kubeflow_pipelines.png" alt="View details about a particular run (and link back to the Kubeflow UI)" >}}

{{< img src="/images/integrations/kubeflow_via_app.png" alt="See the full DAG of inputs and outputs at each stage of your pipeline" >}}

### Via the Public API (for programmatic access)

* For programmatic access, [see our Public API](/ref/python/public-api).

### Concept mapping from Kubeflow Pipelines to W&B

Here's a mapping of Kubeflow Pipelines concepts to W&B

| Kubeflow Pipelines | W&B | Location in W&B |
| ------------------ | --- | --------------- |
| Input Scalar | [`config`](/guides/track/config) | [Overview tab](../../runs/intro.md#overview-tab) |
| Output Scalar | [`summary`](/guides/track/log) | [Overview tab](../../runs/intro.md#overview-tab) |
| Input Artifact | Input Artifact | [Artifacts tab](../../runs/intro.md#artifacts-tab) |
| Output Artifact | Output Artifact | [Artifacts tab](../../runs/intro.md#artifacts-tab) |

## Fine-grain logging

If you want finer control of logging, you can sprinkle in `wandb.log` and `wandb.log_artifact` calls in the component.

### With explicit `wandb.log_artifacts` calls

In this example below, we are training a model. The `@wandb_log` decorator will automatically track the relevant inputs and outputs. If you want to log the training process, you can explicitly add that logging like so:

```python
@wandb_log
def train_model(
    train_dataloader_path: components.InputPath("dataloader"),
    test_dataloader_path: components.InputPath("dataloader"),
    model_path: components.OutputPath("pytorch_model"),
):
    ...
    for epoch in epochs:
        for batch_idx, (data, target) in enumerate(train_dataloader):
            ...
            if batch_idx % log_interval == 0:
                wandb.log(
                    {"epoch": epoch, "step": batch_idx * len(data), "loss": loss.item()}
                )
        ...
        wandb.log_artifact(model_artifact)
```

### With implicit wandb integrations

If you're using a [framework integration we support](/guides/integrations), you can also pass in the callback directly:

```python
@wandb_log
def train_model(
    train_dataloader_path: components.InputPath("dataloader"),
    test_dataloader_path: components.InputPath("dataloader"),
    model_path: components.OutputPath("pytorch_model"),
):
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning import Trainer

    trainer = Trainer(logger=WandbLogger())
    ...  # do training
```