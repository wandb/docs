---
title: Sweeps と SageMaker を使用できますか？
menu:
  support:
    identifier: ja-support-kb-articles-sweeps_sagemaker
support:
- スイープ
- aws
toc_hide: true
type: docs
url: /ja/support/:filename
---

W&B を認証するには、以下の手順を完了してください。Amazon SageMaker の組み込み推定器を使用する場合、`requirements.txt` ファイルを作成します。認証および `requirements.txt` ファイルの設定に関する詳細は、[SageMaker インテグレーション]({{< relref path="/guides/integrations/sagemaker.md" lang="ja" >}}) ガイドを参照してください。

{{% alert %}}
[GitHub](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cifar10-sagemaker) で完全な例を見つけ、[ブログ](https://wandb.ai/site/articles/running-sweeps-with-sagemaker)で追加の洞察を得ることができます。\
SageMaker と W&B を使用して感情分析をデプロイするための[チュートリアル](https://wandb.ai/authors/sagemaker/reports/Deploy-Sentiment-Analyzer-Using-SageMaker-and-W-B--VmlldzoxODA1ODE)にアクセスしてください。
{{% /alert %}}