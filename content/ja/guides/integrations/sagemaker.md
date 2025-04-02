---
title: SageMaker
description: W&B と Amazon SageMaker の統合方法について。
menu:
  default:
    identifier: ja-guides-integrations-sagemaker
    parent: integrations
weight: 370
---

W&B は [Amazon SageMaker](https://aws.amazon.com/sagemaker/) と連携し、ハイパーパラメータの自動読み取り、分散 run のグループ化、チェックポイントからの run の再開を自動で行います。

## 認証

W&B は、トレーニングスクリプトを基準とした `secrets.env` という名前のファイルを探し、`wandb.init()` が呼び出されると、それらを環境にロードします。`secrets.env` ファイルは、実験の launch に使用するスクリプトで `wandb.sagemaker_auth(path="source_dir")` を呼び出すことによって生成できます。このファイルを必ず `.gitignore` に追加してください!

## 既存のエスティメーター

SageMaker の事前構成済みエスティメーターのいずれかを使用している場合は、wandb を含む `requirements.txt` をソースディレクトリーに追加する必要があります。

```text
wandb
```

Python 2 を実行しているエスティメーターを使用している場合は、wandb をインストールする前に、この [wheel](https://pythonwheels.galaxyproject.org/packages/psutil-5.4.8-cp27-cp27mu-manylinux1_x86_64.whl) から直接 `psutil` をインストールする必要があります。

```text
https://wheels.galaxyproject.org/packages/psutil-5.4.8-cp27-cp27mu-manylinux1_x86_64.whl
wandb
```

完全な例は [GitHub](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cifar10-sagemaker) で確認し、詳細については [blog](https://wandb.ai/site/articles/running-sweeps-with-sagemaker) を参照してください。

SageMaker と W&B を使用したセンチメント分析アナライザーのデプロイに関する [チュートリアル](https://wandb.ai/authors/sagemaker/reports/Deploy-Sentiment-Analyzer-Using-SageMaker-and-W-B--VmlldzoxODA1ODE) もお読みいただけます。

{{% alert color="secondary" %}}
W&B の sweep agent は、SageMaker インテグレーションが無効になっている場合にのみ、SageMaker ジョブで期待どおりに動作します。`wandb.init` の呼び出しを変更して、SageMaker インテグレーションをオフにします。

```python
wandb.init(..., settings=wandb.Settings(sagemaker_disable=True))
```
{{% /alert %}}
