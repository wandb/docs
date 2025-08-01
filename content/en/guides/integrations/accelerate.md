---
description: Training and inference at scale made simple, efficient and adaptable
menu:
  default:
    identifier: accelerate
    parent: integrations
title: Hugging Face Accelerate
weight: 140
---

Hugging Face Accelerate is a library that enables the same PyTorch code to run across any distributed configuration, to simplify model training and inference at scale.

Accelerate includes a Weights & Biases Tracker which we show how to use below. You can also read more about [Accelerate Trackers in Hugging Face](https://huggingface.co/docs/accelerate/main/en/usage_guides/tracking).

## Start logging with Accelerate

To get started with Accelerate and Weights & Biases you can follow the pseudocode below:

```python
from accelerate import Accelerator

# Tell the Accelerator object to log with wandb
accelerator = Accelerator(log_with="wandb")

# Initialise your wandb run, passing wandb parameters and any config information
accelerator.init_trackers(
    project_name="my_project", 
    config={"dropout": 0.1, "learning_rate": 1e-2}
    init_kwargs={"wandb": {"entity": "my-wandb-team"}}
    )

...

# Log to wandb by calling `accelerator.log`, `step` is optional
accelerator.log({"train_loss": 1.12, "valid_loss": 0.8}, step=global_step)


# Make sure that the wandb tracker finishes correctly
accelerator.end_training()
```

Explaining more, you need to:
1. Pass `log_with="wandb"` when initialising the Accelerator class
2. Call the [`init_trackers`](https://huggingface.co/docs/accelerate/main/en/package_reference/accelerator#accelerate.Accelerator.init_trackers) method and pass it:
- a project name via `project_name`
- any parameters you want to pass to [`wandb.init()`]({{< relref "/ref/python/sdk/functions/init.md" >}}) via a nested dict to `init_kwargs`
- any other experiment config information you want to log to your wandb run, via `config`
3. Use the `.log` method to log to Weigths & Biases; the `step` argument is optional
4. Call `.end_training` when finished training

## Access the W&B tracker

To access the W&B tracker, use the `Accelerator.get_tracker()` method. Pass in the string corresponding to a tracker’s `.name` attribute, which returns the tracker on the `main` process.

```python
wandb_tracker = accelerator.get_tracker("wandb")

```
From there you can interact with wandb’s run object like normal:

```python
wandb_tracker.log_artifact(some_artifact_to_log)
```

{{% alert color="secondary" %}}
Trackers built in Accelerate will automatically execute on the correct process, so if a tracker is only meant to be ran on the main process it will do so automatically.

If you want to truly remove Accelerate’s wrapping entirely, you can achieve the same outcome with:

```python
wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
with accelerator.on_main_process:
    wandb_tracker.log_artifact(some_artifact_to_log)
```
{{% /alert %}}

## Accelerate Articles
Below is an Accelerate article you may enjoy

<details>

<summary>HuggingFace Accelerate Super Charged With Weights & Biases</summary>

* In this article, we'll look at what HuggingFace Accelerate has to offer and how simple it is to perform distributed training and evaluation, while logging results to Weights & Biases

Read the [Hugging Face Accelerate Super Charged with Weights & Biases report](https://wandb.ai/gladiator/HF%20Accelerate%20+%20W&B/reports/Hugging-Face-Accelerate-Super-Charged-with-Weights-Biases--VmlldzoyNzk3MDUx?utm_source=docs&utm_medium=docs&utm_campaign=accelerate-docs).
</details>
<br /><br />