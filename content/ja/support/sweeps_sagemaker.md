---
title: Can I use Sweeps and SageMaker?
menu:
  support:
    identifier: ja-support-sweeps_sagemaker
tags:
- sweeps
- aws
toc_hide: true
type: docs
---

W&B を認証するには、以下の手順を完了してください。組み込みの Amazon SageMaker エスティメーターを使用する場合は、`requirements.txt` ファイルを作成します。認証と `requirements.txt` ファイルの設定の詳細については、[SageMaker インテグレーション]({{< relref path="/guides/integrations/sagemaker.md" lang="ja" >}}) ガイドを参照してください。

{{% alert %}}
[GitHub](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cifar10-sagemaker) で完全な例を見つけて、さらに多くの洞察を提供する私たちの [ブログ](https://wandb.ai/site/articles/running-sweeps-with-sagemaker) をご覧ください。\
SageMaker と W&B を使用してセンチメントアナライザーをデプロイするための [チュートリアル](https://wandb.ai/authors/sagemaker/reports/Deploy-Sentiment-Analyzer-Using-SageMaker-and-W-B--VmlldzoxODA1ODE) にアクセスしてください。
{{% /alert %}}