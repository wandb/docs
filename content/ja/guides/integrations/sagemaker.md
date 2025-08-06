---
title: SageMaker
description: W&B を Amazon SageMaker と統合する方法
menu:
  default:
    identifier: ja-guides-integrations-sagemaker
    parent: integrations
weight: 370
---

W&B は [Amazon SageMaker](https://aws.amazon.com/sagemaker/) と統合し、ハイパーパラメーターの自動読み込み、分散 run のグルーピング、チェックポイントからの run の再開などを行います。

## 認証

W&B はトレーニングスクリプトの相対パスに配置された `secrets.env` ファイルを探し、`wandb.init()` の呼び出し時にその内容を環境変数として読み込みます。`secrets.env` ファイルは、実験をローンチするためのスクリプト内で `wandb.sagemaker_auth(path="source_dir")` を呼び出すことで生成できます。このファイルは `.gitignore` に追加することを忘れないでください！

## 既存の estimator の利用

SageMaker の標準 estimator を利用している場合は、W&B を含めた `requirements.txt` をソースディレクトリに追加してください。

```text
wandb
```

Python 2 上で estimator を実行している場合は、W&B をインストールする前に [この wheel](https://pythonwheels.com) から `psutil` を直接インストールする必要があります。

```text
https://wheels.galaxyproject.org/packages/psutil-5.4.8-cp27-cp27mu-manylinux1_x86_64.whl
wandb
```

[GitHub](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cifar10-sagemaker) で完全な例を見ることができ、[ブログ](https://wandb.ai/site/articles/running-sweeps-with-sagemaker) でも詳しく解説しています。

また、SageMaker と W&B を使ってセンチメントアナライザーをデプロイするチュートリアルは [Deploy Sentiment Analyzer Using SageMaker and W&B tutorial](https://wandb.ai/authors/sagemaker/reports/Deploy-Sentiment-Analyzer-Using-SageMaker-and-W-B--VmlldzoxODA1ODE) でご覧いただけます。

{{% alert color="secondary" %}}
W&B sweep agent は、SageMaker のインテグレーションが無効になっている場合のみ SageMaker ジョブで期待通り動作します。SageMaker インテグレーションをオフにするには、`wandb.init` の呼び出し時に以下のように設定してください。

```python
wandb.init(..., settings=wandb.Settings(sagemaker_disable=True))
```
{{% /alert %}}