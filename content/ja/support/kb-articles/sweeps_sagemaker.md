---
title: Sweeps と SageMaker を一緒に使えますか？
url: /support/:filename
toc_hide: true
type: docs
support:
- sweeps
- 'aws

  '
---

W&B を認証するには、以下の手順を実行してください。組み込みの Amazon SageMaker estimator を使用する場合は、`requirements.txt` ファイルを作成します。認証や `requirements.txt` ファイルの設定方法については、[SageMaker インテグレーション]({{< relref "/guides/integrations/sagemaker.md" >}}) ガイドをご覧ください。

{{% alert %}}
[GitHub](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cifar10-sagemaker) で完全な例を確認できます。また、[ブログ](https://wandb.ai/site/articles/running-sweeps-with-sagemaker) では追加のインサイトもご紹介しています。\
SageMaker と W&B を使ったセンチメントアナライザーのデプロイ方法は、[Deploy Sentiment Analyzer Using SageMaker and W&B tutorial](https://wandb.ai/authors/sagemaker/reports/Deploy-Sentiment-Analyzer-Using-SageMaker-and-W-B--VmlldzoxODA1ODE) をご覧ください。
{{% /alert %}}