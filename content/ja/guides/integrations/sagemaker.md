---
title: SageMaker
description: W&B と Amazon SageMaker の統合方法。
menu:
  default:
    identifier: ja-guides-integrations-sagemaker
    parent: integrations
weight: 370
---

W&B は [Amazon SageMaker](https://aws.amazon.com/sagemaker/) と連携し、ハイパーパラメーターの自動読み取り、分散 run のグループ化、チェックポイントからの run の再開を自動で行います。

## 認証

W&B は、トレーニング スクリプトを基準とした `secrets.env` という名前のファイルを探し、`wandb.init()` が呼び出されると、それらを環境に読み込みます。`secrets.env` ファイルを生成するには、実験 の起動に使用するスクリプトで `wandb.sagemaker_auth(path="source_dir")` を呼び出します。このファイルを必ず `.gitignore` に追加してください!

## 既存の estimator

SageMaker の事前構成済みの estimator のいずれかを使用している場合は、wandb を含む `requirements.txt` をソース ディレクトリーに追加する必要があります。

```text
wandb
```

Python 2 を実行している estimator を使用している場合は、wandb をインストールする前に、この [wheel](https://pythonwheels.com) から直接 `psutil` をインストールする必要があります。

```text
https://wheels.galaxyproject.org/packages/psutil-5.4.8-cp27-cp27mu-manylinux1_x86_64.whl
wandb
```

[GitHub](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cifar10-sagemaker) で完全な例を確認し、[ブログ](https://wandb.ai/site/articles/running-sweeps-with-sagemaker) で詳細をご覧ください。

SageMaker と W&B を使用してセンチメント分析器をデプロイする方法については、[チュートリアル](https://wandb.ai/authors/sagemaker/reports/Deploy-Sentiment-Analyzer-Using-SageMaker-and-W-B--VmlldzoxODA1ODE) をお読みください。

{{% alert color="secondary" %}}
W&B sweep agent は、SageMaker のインテグレーションがオフになっている場合にのみ、SageMaker ジョブで期待どおりに動作します。`wandb.init` の呼び出しを変更して、SageMaker のインテグレーションをオフにします。

```python
wandb.init(..., settings=wandb.Settings(sagemaker_disable=True))
```
{{% /alert %}}
