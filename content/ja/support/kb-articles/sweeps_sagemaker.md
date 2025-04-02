---
title: Can I use Sweeps and SageMaker?
menu:
  support:
    identifier: ja-support-kb-articles-sweeps_sagemaker
support:
- sweeps
- aws
toc_hide: true
type: docs
url: /support/:filename
---

W&B を認証するには、以下の手順を実行してください。組み込みの Amazon SageMaker estimator を使用している場合は、`requirements.txt` ファイルを作成します。認証と `requirements.txt` ファイルのセットアップの詳細については、[SageMaker integration]({{< relref path="/guides/integrations/sagemaker.md" lang="ja" >}}) ガイドを参照してください。

{{% alert %}}
完全な例は [GitHub](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cifar10-sagemaker) で確認できます。また、[blog](https://wandb.ai/site/articles/running-sweeps-with-sagemaker) で追加の洞察を得られます。\
SageMaker と W&B を使用してセンチメント分析器をデプロイするための [tutorial](https://wandb.ai/authors/sagemaker/reports/Deploy-Sentiment-Analyzer-Using-SageMaker-and-W-B--VmlldzoxODA1ODE) にアクセスしてください。
{{% /alert %}}
