---
cascade:
- url: guides/track/log/:filename
description: Keep track of metrics, videos, custom plots, and more
menu:
  default:
    identifier: ko-guides-models-track-log-_index
    parent: experiments
title: Log objects and media
url: guides/track/log
weight: 6
---

Log a dictionary of metrics, media, or custom objects to a step with the W&B Python SDK. W&B collects the key-value pairs during each step and stores them in one unified dictionary each time you log data with `wandb.log()`. Data logged from your script is saved locally to your machine in a directory called `wandb`, then synced to the W&B cloud or your [private server]({{< relref path="/guides/hosting/" lang="ko" >}}). 

{{% alert %}}
Key-value pairs are stored in one unified dictionary only if you pass the same value for each step. W&B writes all of the collected keys and values to memory if you log a different value for `step`.
{{% /alert %}}

Each call to `wandb.log` is a new `step` by default. W&B uses steps as the default x-axis when it creates charts and panels. You can optionally create and use a custom x-axis or capture a custom summary metric. For more information, see [Customize log axes]({{< relref path="./customize-logging-axes.md" lang="ko" >}}).

<!-- [INSERT BETTER EXAMPLE] -->
<!-- If you want to log to a single history step from lots of different places in your code you can pass a step index to `wandb.log()` as follows:

```python
wandb.log({'loss': 0.2}, step=step)
``` -->

<!-- [INSERT EXAMPLE] -->

{{% alert color="secondary" %}}
Use `wandb.log()` to log consecutive values for each `step`: 0, 1, 2, and so on. It is not possible to write to a specific history step. W&B only writes to the "current" and "next" step.
{{% /alert %}}

<!-- You can set `commit=False` in `wandb.log` to accumulate metrics, just be sure to eventually call `wandb.log` with `commit=True` (the default) to persist the metrics.

```python
wandb.log({'loss': 0.2}, commit=False)
# Somewhere else when I'm ready to report this step:
wandb.log({'accuracy': 0.8})
``` -->


## Automatically logged data

W&B automatically logs the following information during a W&B Experiment:


* **System metrics**: CPU and GPU utilization, network, etc. These are shown in the System tab on the [run page]({{< relref path="/guides/models/track/runs/" lang="ko" >}}). For the GPU, these are fetched with [`nvidia-smi`](https://developer.nvidia.com/nvidia-system-management-interface).
* **Command line**: The stdout and stderr are picked up and show in the logs tab on the [run page.]({{< relref path="/guides/models/track/runs/" lang="ko" >}})

Turn on [Code Saving](http://wandb.me/code-save-colab) in your account's [Settings page](https://wandb.ai/settings) to log:

* **Git commit**: Pick up the latest git commit and see it on the overview tab of the run page, as well as a `diff.patch` file if there are any uncommitted changes.
* **Dependencies**: The `requirements.txt` file will be uploaded and shown on the files tab of the run page, along with any files you save to the `wandb` directory for the run.


## What data is logged with specific W&B API calls?

With W&B, you can decide exactly what you want to log. The following lists some commonly logged objects:

* **Datasets**: You have to specifically log images or other dataset samples for them to stream to W&B.
* **Plots**: Use `wandb.plot` with `wandb.log` to track charts. See [Log Plots]({{< relref path="./plots.md" lang="ko" >}}) for more information. 
* **Tables**: Use `wandb.Table` to log data to visualize and query with W&B. See [Log Tables]({{< relref path="./log-tables.md" lang="ko" >}}) for more information.
* **PyTorch gradients**: Add `wandb.watch(model)` to see gradients of the weights as histograms in the UI.
* **Configuration information**: Log hyperparameters, a link to your dataset, or the name of the architecture you're using as config parameters, passed in like this: `wandb.init(config=your_config_dictionary)`. See the [PyTorch Integrations]({{< relref path="/guides/integrations/pytorch.md" lang="ko" >}}) page for more information. 
* **Metrics**: Use `wandb.log` to see metrics from your model. If you log metrics like accuracy and loss from inside your training loop, you'll get live updating graphs in the UI.

<!-- ### Example Usage

```python
wandb.log({"loss": 0.314, "epoch": 5,
           "inputs": wandb.Image(inputs),
           "logits": wandb.Histogram(outputs),
           "captions": wandb.Html(captions)})
``` -->


## Common workflows

1. **Compare the best accuracy**: To compare the best value of a metric across runs, set the summary value for that metric. By default, summary is set to the last value you logged for each key. This is useful in the table in the UI, where you can sort and filter runs based on their summary metrics, to help compare runs in a table or bar chart based on their _best_ accuracy, instead of final accuracy. For example: `wandb.run.summary["best_accuracy"] = best_accuracy`
2. **View multiple metrics on one chart**: Log multiple metrics in the same call to `wandb.log`, like this: `wandb.log({"acc'": 0.9, "loss": 0.1})` and they will both be available to plot against in the UI
3. **Customize the x-axis**: Add a custom x-axis to the same log call to visualize your metrics against a different axis in the W&B dashboard. For example: `wandb.log({'acc': 0.9, 'epoch': 3, 'batch': 117})`. To set the default x-axis for a given metric use [Run.define_metric()]({{< relref path="/ref/python/run.md#define_metric" lang="ko" >}})
4. **Log rich media and charts**: `wandb.log` supports the logging of a wide variety of data types, from [media like images and videos]({{< relref path="./media.md" lang="ko" >}}) to [tables]({{< relref path="./log-tables.md" lang="ko" >}}) and [charts]({{< relref path="/guides/models/app/features/custom-charts/" lang="ko" >}}).

## Best practices and tips 

For best practices and tips for Experiments and logging, see [Best Practices: Experiments and Logging](https://wandb.ai/wandb/pytorch-lightning-e2e/reports/W-B-Best-Practices-Guide--VmlldzozNTU1ODY1#w&b-experiments-and-logging).