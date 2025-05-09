---
title: SageMaker
description: W&B を Amazon SageMaker と統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-sagemaker
    parent: integrations
weight: 370
---

W&B は [Amazon SageMaker](https://aws.amazon.com/sagemaker/) とインテグレーションしており、ハイパーパラメーターを自動で読み取り、分散 run をグループ化し、チェックポイントから run を再開します。

## 認証

W&B はトレーニングスクリプトと相対的な位置にある `secrets.env` という名前のファイルを探し、`wandb.init()` が呼び出されたときにそれを環境にロードします。`wandb.sagemaker_auth(path="source_dir")` を実行することで、`secrets.env` ファイルを生成できます。このファイルを `.gitignore` に追加することを忘れないでください！

## 既存の推定器

SageMaker の事前設定された推定器を使用している場合、ソースディレクトリーに wandb を含む `requirements.txt` を追加する必要があります。

```text
wandb
```

Python 2 を実行している推定器を使用している場合、wandb をインストールする前に [wheel](https://pythonwheels.com) から直接 `psutil` をインストールする必要があります。

```text
https://wheels.galaxyproject.org/packages/psutil-5.4.8-cp27-cp27mu-manylinux1_x86_64.whl
wandb
```

[GitHub](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cifar10-sagemaker) で完全な例を確認し、[ブログ](https://wandb.ai/site/articles/running-sweeps-with-sagemaker) でさらに詳しく読んでください。

また、SageMaker と W&B を使用した感情分析器のデプロイに関する[チュートリアル](https://wandb.ai/authors/sagemaker/reports/Deploy-Sentiment-Analyzer-Using-SageMaker-and-W-B--VmlldzoxODA1ODE)を読むこともできます。

{{% alert color="secondary" %}}
W&B sweep agent は SageMaker インテグレーションがオフになっている場合のみ SageMaker ジョブで期待通りに動作します。`wandb.init` の呼び出しを変更して SageMaker インテグレーションをオフにしてください。

```python
wandb.init(..., settings=wandb.Settings(sagemaker_disable=True))
```
{{% /alert %}}