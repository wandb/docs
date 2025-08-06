---
title: SageMaker
description: W&B を Amazon SageMaker と統合する方法
menu:
  default:
    identifier: sagemaker
    parent: integrations
weight: 370
---

W&B は [Amazon SageMaker](https://aws.amazon.com/sagemaker/) と連携し、ハイパーパラメーターの自動読み取り、分散 run のグルーピング、チェックポイントからの run の再開を行います。

## 認証

W&B はトレーニングスクリプトの相対パスにある `secrets.env` というファイルを探し、`wandb.init()` が呼び出された際にその内容を環境変数へロードします。`secrets.env` ファイルは、experiment を実行するスクリプト内で `wandb.sagemaker_auth(path="source_dir")` を呼び出すことで生成できます。このファイルは `.gitignore` に必ず追加してください！

## 既存の estimator

SageMaker のプリセット estimator を利用している場合は、source ディレクトリー内に wandb を含む `requirements.txt` を追加する必要があります。

```text
wandb
```

Python 2 が実行される estimator を使用している場合は、wandb をインストールする前に [この wheel](https://pythonwheels.com) から直接 `psutil` をインストールする必要があります。

```text
https://wheels.galaxyproject.org/packages/psutil-5.4.8-cp27-cp27mu-manylinux1_x86_64.whl
wandb
```

[GitHub](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cifar10-sagemaker) で完全なサンプルコードを確認したり、[ブログ](https://wandb.ai/site/articles/running-sweeps-with-sagemaker) で詳細を読むことができます。

また、「[Deploy Sentiment Analyzer Using SageMaker and W&B tutorial](https://wandb.ai/authors/sagemaker/reports/Deploy-Sentiment-Analyzer-Using-SageMaker-and-W-B--VmlldzoxODA1ODE)」で、SageMaker と W&B を使ったセンチメントアナライザーのデプロイ手順もご覧いただけます。

{{% alert color="secondary" %}}
W&B sweep agent は、SageMaker のインテグレーションが無効化されている場合のみ SageMaker ジョブ上で正しく動作します。`wandb.init` を呼び出す際に SageMaker のインテグレーションをオフにしてください。

```python
wandb.init(..., settings=wandb.Settings(sagemaker_disable=True))
```
{{% /alert %}}