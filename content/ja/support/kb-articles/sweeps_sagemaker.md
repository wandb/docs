---
title: Sweeps と SageMaker を一緒に使えますか？
menu:
  support:
    identifier: ja-support-kb-articles-sweeps_sagemaker
support:
- スイープ
- aws
toc_hide: true
type: docs
url: /support/:filename
---

W&B を認証するには、以下の手順を実行してください。Amazon SageMaker の組み込み Estimator を使用する場合は、`requirements.txt` ファイルを作成します。認証や `requirements.txt` ファイルの設定方法については、[SageMaker インテグレーション]({{< relref path="/guides/integrations/sagemaker.md" lang="ja" >}}) ガイドをご参照ください。

{{% alert %}}
[GitHub](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cifar10-sagemaker) には完全なサンプルがあります。また、[ブログ](https://wandb.ai/site/articles/running-sweeps-with-sagemaker) でも追加の知見が得られます。  
SageMaker と W&B を使ってセンチメント分析器をデプロイする方法については、[「Deploy Sentiment Analyzer Using SageMaker and W&B」チュートリアル](https://wandb.ai/authors/sagemaker/reports/Deploy-Sentiment-Analyzer-Using-SageMaker-and-W-B--VmlldzoxODA1ODE) をご覧ください。
{{% /alert %}}