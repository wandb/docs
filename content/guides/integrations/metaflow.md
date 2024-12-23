---
description: How to integrate W&B with Metaflow.
menu:
  default:
    identifier: metaflow
    parent: integrations
title: Metaflow
weight: 200
---
## Overview

[Metaflow](https://docs.metaflow.org) is a framework created by [Netflix](https://netflixtechblog.com) for creating and running ML workflows.

This integration lets users apply decorators to Metaflow [steps and flows](https://docs.metaflow.org/metaflow/basics) to automatically log parameters and artifacts to W&B.

* Decorating a step will turn logging off or on for certain types within that step.
* Decorating the flow will turn logging off or on for every step in the flow.

## Quickstart

### Install W&B and login

{{< tabpane text=true >}}

{{% tab header="Notebook" value="notebook" %}}

```python
!pip install -Uqqq metaflow fastcore wandb

import wandb
wandb.login()
```

{{% /tab %}}

{{% tab header="Command Line" value="cli" %}}

```
pip install -Uqqq metaflow fastcore wandb
wandb login
```

{{% /tab %}}

{{< /tabpane >}}

### Decorate your flows and steps

{{< tabpane text=true >}}
{{% tab header="Step" value="step" %}}

Decorating a step will turn logging off or on for certain types within that Step.

In this example, all datasets and models in `start` will be logged

```python
from wandb.integration.metaflow import wandb_log

class WandbExampleFlow(FlowSpec):
    @wandb_log(datasets=True, models=True, settings=wandb.Settings(...))
    @step
    def start(self):
        self.raw_df = pd.read_csv(...).    # pd.DataFrame -> upload as dataset
        self.model_file = torch.load(...)  # nn.Module    -> upload as model
        self.next(self.transform)
```
{{% /tab %}}

{{% tab header="Flow" value="flow" %}}

Decorating a flow is equivalent to decorating all the constituent steps with a default.

In this case, all steps in `WandbExampleFlow` will log datasets and models by default -- the same as decorating each step with `@wandb_log(datasets=True, models=True)`

```python
from wandb.integration.metaflow import wandb_log

@wandb_log(datasets=True, models=True)  # decorate all @step 
class WandbExampleFlow(FlowSpec):
    @step
    def start(self):
        self.raw_df = pd.read_csv(...).    # pd.DataFrame -> upload as dataset
        self.model_file = torch.load(...)  # nn.Module    -> upload as model
        self.next(self.transform)
```
{{% /tab %}}

{{% tab header="Flow and Steps" value="flow_and_steps" %}}

Decorating the flow is equivalent to decorating all steps with a default. That means if you later decorate a Step with another `@wandb_log`, you will override the flow-level decoration.

In the example below:

* `start` and `mid` will log datasets and models, but
* `end` will not log datasets or models.

```python
from wandb.integration.metaflow import wandb_log

@wandb_log(datasets=True, models=True)  # same as decorating start and mid
class WandbExampleFlow(FlowSpec):
  # this step will log datasets and models
  @step
  def start(self):
    self.raw_df = pd.read_csv(...).    # pd.DataFrame -> upload as dataset
    self.model_file = torch.load(...)  # nn.Module    -> upload as model
    self.next(self.mid)

  # this step will also log datasets and models
  @step
  def mid(self):
    self.raw_df = pd.read_csv(...).    # pd.DataFrame -> upload as dataset
    self.model_file = torch.load(...)  # nn.Module    -> upload as model
    self.next(self.end)

  # this step is overwritten and will NOT log datasets OR models
  @wandb_log(datasets=False, models=False)
  @step
  def end(self):
    self.raw_df = pd.read_csv(...).    
    self.model_file = torch.load(...)
```
{{% /tab %}}
{{< /tabpane >}}

## Where is my data? Can I access it programmatically?

You can access the information we've captured in three ways: inside the original Python process being logged using the [`wandb` client library](../../../ref/python/README.md), with the [web app UI](../../track/workspaces.md), or programmatically using [our Public API](../../../ref/python/public-api/README.md). `Parameter`s are saved to W&B's [`config`](../../track/config.md) and can be found in the [Overview tab](../../runs/intro.md#overview-tab). `datasets`, `models`, and `others` are saved to [W&B Artifacts](../../artifacts/intro.md) and can be found in the [Artifacts tab](../../runs/intro.md#artifacts-tab). Base python types are saved to W&B's [`summary`](../../track/log/intro.md) dict and can be found in the Overview tab. See our [guide to the Public API](../../track/public-api-guide.md) for details on using the API to get this information programmatically from outside .

Here's a cheatsheet:

| Data                                            | Client library                            | UI                    |
| ----------------------------------------------- | ----------------------------------------- | --------------------- |
| `Parameter(...)`                                | `wandb.config`                            | Overview tab, Config  |
| `datasets`, `models`, `others`                  | `wandb.use_artifact("{var_name}:latest")` | Artifacts tab         |
| Base Python types (`dict`, `list`, `str`, etc.) | `wandb.summary`                           | Overview tab, Summary |

### `wandb_log` kwargs

| kwarg      | Options                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `datasets` | <ul><li><code>True</code>: Log instance variables that are a dataset</li><li><code>False</code></li></ul>                                                                                                                                                                                                                                                                                                                                                                         |
| `models`   | <ul><li><code>True</code>: Log instance variables that are a model</li><li><code>False</code></li></ul>                                                                                                                                                                                                                                                                                                                                                                           |
| `others`   | <ul><li><code>True</code>: Log anything else that is serializable as a pickle</li><li><code>False</code></li></ul>                                                                                                                                                                                                                                                                                                                                                                |
| `settings` | <ul><li><code>wandb.Settings(...)</code>: Specify your own <code>wandb</code> settings for this step or flow</li><li><code>None</code>: Equivalent to passing <code>wandb.Settings()</code></li></ul><p>By default, if:</p><ul><li><code>settings.run_group</code> is <code>None</code>, it will be set to <code>\{flow_name\}/\{run_id\}</code></li><li><code>settings.run_job_type</code> is <code>None</code>, it will be set to <code>\{run_job_type\}/\{step_name\}</code></li></ul> |

## Frequently Asked Questions

### What exactly do you log? Do you log all instance and local variables?

`wandb_log` only logs instance variables. Local variables are NEVER logged. This is useful to avoid logging unnecessary data.

### Which data types get logged?

We currently support these types:

| Logging Setting     | Type                                                                                                                        |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| default (always on) | <ul><li><code>dict, list, set, str, int, float, bool</code></li></ul>                                                       |
| `datasets`          | <ul><li><code>pd.DataFrame</code></li><li><code>pathlib.Path</code></li></ul>                                               |
| `models`            | <ul><li><code>nn.Module</code></li><li><code>sklearn.base.BaseEstimator</code></li></ul>                                    |
| `others`            | <ul><li>Anything that is <a href="https://wiki.python.org/moin/UsingPickle">pickle-able</a> and JSON serializable</li></ul> |

### Examples of logging behavior

| Kind of Variable | behavior                      | Example         | Data Type      |
| ---------------- | ------------------------------ | --------------- | -------------- |
| Instance         | Auto-logged                    | `self.accuracy` | `float`        |
| Instance         | Logged if `datasets=True`      | `self.df`       | `pd.DataFrame` |
| Instance         | Not logged if `datasets=False` | `self.df`       | `pd.DataFrame` |
| Local            | Never logged                   | `accuracy`      | `float`        |
| Local            | Never logged                   | `df`            | `pd.DataFrame` |

### Does this track artifact lineage?

Yes! If you have an artifact that is an output of step A and an input to step B, we automatically construct the lineage DAG for you.

For an example of this behavior, please see this[ notebook](https://colab.research.google.com/drive/1wZG-jYzPelk8Rs2gIM3a71uEoG46u_nG#scrollTo=DQQVaKS0TmDU) and its corresponding [W&B Artifacts page](https://wandb.ai/megatruong/metaflow_integration/artifacts/dataset/raw_df/7d14e6578d3f1cfc72fe/graph)