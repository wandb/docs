---
title: Sweeps と SageMaker は 併用できますか？
menu:
  support:
    identifier: ja-support-kb-articles-sweeps_sagemaker
support:
- sweeps
- AWS
toc_hide: true
type: docs
url: /support/:filename
---

W&B を認証するには、次を実行します。組み込みの Amazon SageMaker Estimator を使用する場合は `requirements.txt` ファイルを作成します。認証と `requirements.txt` ファイルの設定の詳細は、[SageMaker インテグレーション]({{< relref path="/guides/integrations/sagemaker.md" lang="ja" >}}) ガイドを参照してください。

{{% alert %}}
完全な例は [GitHub](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cifar10-sagemaker) に、さらに詳しい解説は当社の [ブログ](https://wandb.ai/site/articles/running-sweeps-with-sagemaker) にあります。\
SageMaker と W&B を使って感情分析器をデプロイするには、[SageMaker と W&B で感情分析器をデプロイするチュートリアル](https://wandb.ai/authors/sagemaker/reports/Deploy-Sentiment-Analyzer-Using-SageMaker-and-W-B--VmlldzoxODA1ODE) にアクセスしてください。
{{% /alert %}}