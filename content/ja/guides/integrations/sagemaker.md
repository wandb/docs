---
title: SageMaker
description: W&B を Amazon SageMaker と連携する方法。
menu:
  default:
    identifier: ja-guides-integrations-sagemaker
    parent: integrations
weight: 370
---

W&B は [Amazon SageMaker](https://aws.amazon.com/sagemaker/) と連携し、ハイパーパラメーターの自動読み取り、分散 run のグルーピング、チェックポイントからの run の再開を行います。

## 認証

W&B は トレーニングスクリプト からの相対パスにある `secrets.env` というファイルを探し、`wandb.init()` が呼ばれたタイミングでその内容を 環境 に読み込みます。実験 を起動するための スクリプト 内で `wandb.sagemaker_auth(path="source_dir")` を呼び出すと、`secrets.env` ファイルを生成できます。このファイルは必ず `.gitignore` に追加してください！

## 既存の Estimator

SageMaker のあらかじめ構成された Estimator のいずれかを使用している場合は、wandb を含めた `requirements.txt` をソース ディレクトリーに追加する必要があります。

```text
wandb
```

Python 2 で動作している Estimator を使用している場合は、wandb をインストールする前に、この [wheel](https://pythonwheels.com) から直接 `psutil` をインストールしてください。

```text
https://wheels.galaxyproject.org/packages/psutil-5.4.8-cp27-cp27mu-manylinux1_x86_64.whl
wandb
```

完全なサンプルは [GitHub](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cifar10-sagemaker) をご覧ください。詳しくは当社の [ブログ](https://wandb.ai/site/articles/running-sweeps-with-sagemaker) も参照してください。

また、SageMaker と W&B を使って感情分析器をデプロイする方法については、[Deploy Sentiment Analyzer Using SageMaker and W&B tutorial](https://wandb.ai/authors/sagemaker/reports/Deploy-Sentiment-Analyzer-Using-SageMaker-and-W-B--VmlldzoxODA1ODE) もお読みいただけます。

{{% alert color="secondary" %}}
W&B の sweep agent が SageMaker のジョブで期待どおりに動作するのは、SageMaker インテグレーションを無効にしている場合のみです。`wandb.init` の呼び出しを次のように変更して、SageMaker インテグレーションを無効化してください。

```python
wandb.init(..., settings=wandb.Settings(sagemaker_disable=True))
```
{{% /alert %}}