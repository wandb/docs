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

### Sign up and create an API key

An API key authenticates your machine to W&B. You can generate an API key from your user profile.

{{% alert %}}
For a more streamlined approach, you can generate an API key by going directly to [https://wandb.ai/authorize](https://wandb.ai/authorize). Copy the displayed API key and save it in a secure location such as a password manager.
{{% /alert %}}

1. Click your user profile icon in the upper right corner.
1. Select **User Settings**, then scroll to the **API Keys** section.
1. Click **Reveal**. Copy the displayed API key. To hide the API key, reload the page.

### Install the `wandb` library and log in

To install the `wandb` library locally and log in:

{{% alert %}}
For `wandb` version 0.19.8 or below, install `fastcore` version 1.8.0 or below (`fastcore<1.8.0`) instead of `plum-dispatch`.
{{% /alert %}}


{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. Set the `WANDB_API_KEY` [environment variable]({{< relref "/guides/models/track/environment-variables.md" >}}) to your API key.

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. Install the `wandb` library and log in.



    ```shell
    pip install -Uqqq metaflow "plum-dispatch<3.0.0" wandb

    wandb login
    ```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```bash
pip install -Uqqq metaflow "plum-dispatch<3.0.0" wandb
```
```python
import wandb
wandb.login()
```

{{% /tab %}}

{{% tab header="Python notebook" value="notebook" %}}

```notebook
!pip install -Uqqq metaflow "plum-dispatch<3.0.0" wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

### Decorate your flows and steps

{{< tabpane text=true >}}
{{% tab header="Step" value="step" %}}

Decorating a step turns logging off or on for certain types within that step.

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

In this case, all steps in `WandbExampleFlow` default to logging datasets and models by default, just like decorating each step with `@wandb_log(datasets=True, models=True)`

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

Decorating the flow is equivalent to decorating all steps with a default. That means if you later decorate a Step with another `@wandb_log`, it overrides the flow-level decoration.

In this example:

* `start` and `mid` log both datasets and models.
* `end` logs neither datasets nor models.

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

## Access your data programmatically

You can access the information we've captured in three ways: inside the original Python process being logged using the [`wandb` client library]({{< relref "/ref/python/" >}}), with the [web app UI]({{< relref "/guides/models/track/workspaces.md" >}}), or programmatically using [our Public API]({{< relref "/ref/python/public-api/" >}}). `Parameter`s are saved to W&B's [`config`]({{< relref "/guides/models/track/config.md" >}}) and can be found in the [Overview tab]({{< relref "/guides/models/track/runs/#overview-tab" >}}). `datasets`, `models`, and `others` are saved to [W&B Artifacts]({{< relref "/guides/core/artifacts/" >}}) and can be found in the [Artifacts tab]({{< relref "/guides/models/track/runs/#artifacts-tab" >}}). Base python types are saved to W&B's [`summary`]({{< relref "/guides/models/track/log/" >}}) dict and can be found in the Overview tab. See our [guide to the Public API]({{< relref "/guides/models/track/public-api-guide.md" >}}) for details on using the API to get this information programmatically from outside .

### Quick reference

| Data                                            | Client library                            | UI                    |
| ----------------------------------------------- | ----------------------------------------- | --------------------- |
| `Parameter(...)`                                | `wandb.Run.config`                            | Overview tab, Config  |
| `datasets`, `models`, `others`                  | `wandb.Run.use_artifact("{var_name}:latest")` | Artifacts tab         |
| Base Python types (`dict`, `list`, `str`, etc.) | `wandb.Run.summary`                           | Overview tab, Summary |

### `wandb_log` kwargs

| kwarg      | Options                                                                         |
| ---------- | ------------------------------------------------------------------------------- |
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

### How can I configure logging behavior?

| Kind of Variable | behavior                      | Example         | Data Type      |
| ---------------- | ------------------------------ | --------------- | -------------- |
| Instance         | Auto-logged                    | `self.accuracy` | `float`        |
| Instance         | Logged if `datasets=True`      | `self.df`       | `pd.DataFrame` |
| Instance         | Not logged if `datasets=False` | `self.df`       | `pd.DataFrame` |
| Local            | Never logged                   | `accuracy`      | `float`        |
| Local            | Never logged                   | `df`            | `pd.DataFrame` |

### Is artifact lineage tracked?

Yes. If you have an artifact that is an output of step A and an input to step B, we automatically construct the lineage DAG for you.

For an example of this behavior, please see this[ notebook](https://colab.research.google.com/drive/1wZG-jYzPelk8Rs2gIM3a71uEoG46u_nG#scrollTo=DQQVaKS0TmDU) and its corresponding [W&B Artifacts page](https://wandb.ai/megatruong/metaflow_integration/artifacts/dataset/raw_df/7d14e6578d3f1cfc72fe/graph)