---
title: "Can I use Sweeps and SageMaker?"
tags:
   - sweeps
---

Yes. At a glance, you will need to need to authenticate W&B and you will need to create a `requirements.txt` file if you use a built-in SageMaker estimator. For more on how to authenticate and set up a requirements.txt file, see the [SageMaker integration](../integrations/other/sagemaker.md) guide.

:::info
A complete example is available on [GitHub](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cifar10-sagemaker) and you can read more on our [blog](https://wandb.ai/site/articles/running-sweeps-with-sagemaker).\
You can also read the [tutorial](https://wandb.ai/authors/sagemaker/reports/Deploy-Sentiment-Analyzer-Using-SageMaker-and-W-B--VmlldzoxODA1ODE) on deploying a sentiment analyzer using SageMaker and W&B.
:::