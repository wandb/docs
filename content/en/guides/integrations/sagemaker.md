---
description: How to integrate W&B with Amazon SageMaker.
menu:
  default:
    identifier: sagemaker
    parent: integrations
title: SageMaker
weight: 370
---


W&B integrates with [Amazon SageMaker](https://aws.amazon.com/sagemaker/), automatically reading hyperparameters, grouping distributed runs, and resuming runs from checkpoints.

## Authentication

W&B looks for a file named `secrets.env` relative to the training script and loads them into the environment when `wandb.init()` is called. You can generate a `secrets.env` file by calling `wandb.sagemaker_auth(path="source_dir")` in the script you use to launch your experiments. Be sure to add this file to your `.gitignore`!

## Existing estimators

If you're using one of SageMakers preconfigured estimators you need to add a `requirements.txt` to your source directory that includes wandb

```text
wandb
```

If you're using an estimator that's running Python 2, you'll need to install `psutil` directly from this [wheel](https://pythonwheels.com) before installing wandb:

```text
https://wheels.galaxyproject.org/packages/psutil-5.4.8-cp27-cp27mu-manylinux1_x86_64.whl
wandb
```

Review a complete example on [GitHub](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cifar10-sagemaker), and read more on our [blog](https://wandb.ai/site/articles/running-sweeps-with-sagemaker).

You can also read the [Deploy Sentiment Analyzer Using SageMaker and W&B tutorial](https://wandb.ai/authors/sagemaker/reports/Deploy-Sentiment-Analyzer-Using-SageMaker-and-W-B--VmlldzoxODA1ODE) on deploying a sentiment analyzer using SageMaker and W&B.

{{% alert color="secondary" %}}
The W&B sweep agent behaves as expected in a SageMaker job only if your SageMaker integration is turned off. Turn off the SageMaker integration by modifying your invocation of `wandb.init`:

```python
wandb.init(..., settings=wandb.Settings(sagemaker_disable=True))
```
{{% /alert %}}
