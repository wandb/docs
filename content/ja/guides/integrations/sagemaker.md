---
title: SageMaker
description: Amazon SageMaker と W&B を統合する方法.
menu:
  default:
    identifier: ja-guides-integrations-sagemaker
    parent: integrations
weight: 370
---

W&B は [Amazon SageMaker](https://aws.amazon.com/sagemaker/) と連携し、ハイパーパラメーターを自動的に読み込み、分散 run をグループ化し、チェックポイントから run を再開します。

## 認証

W&B はトレーニングスクリプトの相対パスにある `secrets.env` という名前のファイルを探し、`wandb.init()` が呼び出されたときにそれらを環境にロードします。実験をローンンチする際に使用するスクリプトで `wandb.sagemaker_auth(path="source_dir")` を呼び出すことによって `secrets.env` ファイルを生成できます。このファイルを `.gitignore` に追加することを忘れないでください！

## 既存の推定器

SageMaker の事前構成された推定器の1つを使用している場合は、wandb を含む `requirements.txt` をソースディレクトリに追加する必要があります。

```text
wandb
```

Python 2 を実行している推定器を使用している場合は、wandb をインストールする前に [wheel](https://pythonwheels.com) から直接 `psutil` をインストールする必要があります。

```text
https://wheels.galaxyproject.org/packages/psutil-5.4.8-cp27-cp27mu-manylinux1_x86_64.whl
wandb
```

[GitHub](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cifar10-sagemaker) で完全な例を確認し、私たちの [ブログ](https://wandb.ai/site/articles/running-sweeps-with-sagemaker) をもっと読んでください。

SageMaker と W&B を使用してセンチメントアナライザーをデプロイする [チュートリアル](https://wandb.ai/authors/sagemaker/reports/Deploy-Sentiment-Analyzer-Using-SageMaker-and-W-B--VmlldzoxODA1ODE) もご覧いただけます。

{{% alert color="secondary" %}}
W&B sweep エージェントは、SageMaker インテグレーションがオフになっている場合にのみ、SageMaker ジョブで期待通りに動作します。`wandb.init` の呼び出しを修正して、SageMaker インテグレーションをオフにします。

```python
wandb.init(..., settings=wandb.Settings(sagemaker_disable=True))
```
{{% /alert %}}