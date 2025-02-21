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

W&B を認証するには、次の手順を実行します。組み込みの Amazon SageMaker estimator を使用する場合は、`requirements.txt` ファイルを作成します。認証と `requirements.txt` ファイルのセットアップの詳細については、[SageMaker インテグレーション]({{< relref path="/guides/integrations/sagemaker.md" lang="ja" >}}) ガイドを参照してください。

{{% alert %}}
完全な例は [GitHub](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cifar10-sagemaker) 、および追加の洞察は [ブログ](https://wandb.ai/site/articles/running-sweeps-with-sagemaker) で確認してください。

SageMaker と W&B を使用してセンチメント分析をデプロイするための [チュートリアル](https://wandb.ai/authors/sagemaker/reports/Deploy-Sentiment-Analyzer-Using-SageMaker-and-W-B--VmlldzoxODA1ODE) にアクセスしてください。
{{% /alert %}}
