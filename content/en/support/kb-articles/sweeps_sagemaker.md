---
url: /support/:filename
title: "Can I use Sweeps and SageMaker?"
toc_hide: true
type: docs
support:
   - sweeps
   - aws
---
To authenticate W&B, complete the following steps: create a `requirements.txt` file if using a built-in Amazon SageMaker estimator. For details on authentication and setting up the `requirements.txt` file, refer to the [SageMaker integration]({{< relref "/guides/integrations/sagemaker.md" >}}) guide.

{{% alert %}}
Find a complete example on [GitHub](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cifar10-sagemaker) and additional insights on our [blog](https://wandb.ai/site/articles/running-sweeps-with-sagemaker).\
Access the [Deploy Sentiment Analyzer Using SageMaker and W&B tutorial](https://wandb.ai/authors/sagemaker/reports/Deploy-Sentiment-Analyzer-Using-SageMaker-and-W-B--VmlldzoxODA1ODE) for deploying a sentiment analyzer using SageMaker and W&B.
{{% /alert %}}