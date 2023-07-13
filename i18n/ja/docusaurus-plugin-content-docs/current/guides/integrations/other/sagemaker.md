---
slug: /guides/integrations/sagemaker
description: How to integrate W&B with Amazon SageMaker.
displayed_sidebar: ja
---

# SageMaker

## SageMaker Integration

W&Bは、[Amazon SageMaker](https://aws.amazon.com/sagemaker/)と統合し、ハイパーパラメーターの自動読み取り、分散runのグループ化、チェックポイントからのrunの再開が可能です。

### 認証

W&Bは、トレーニングスクリプトに関連する`secrets.env`という名前のファイルを探し、`wandb.init()`が呼ばれるときに環境にロードします。実験を開始するスクリプトで`wandb.sagemaker_auth(path="source_dir")`を呼び出すことで`secrets.env`ファイルを生成できます。このファイルを`.gitignore`に追加してください。

### 既存のEstimators

SageMakerの事前設定済みestimatorを使用している場合、wandbを含む`requirements.txt`をソースディレクトリに追加する必要があります。

```
wandb
```

Python 2を実行しているestimatorを使用している場合は、wandbをインストールする前に、[wheel](https://pythonwheels.com)からpsutilを直接インストールする必要があります。

```
https://wheels.galaxyproject.org/packages/psutil-5.4.8-cp27-cp27mu-manylinux1_x86_64.whl
wandb
```

:::info
[GitHub](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cifar10-sagemaker)で完全な例が利用できますし、[ブログ](https://wandb.ai/site/articles/running-sweeps-with-sagemaker)でさらに詳細を読めます。\
また、SageMakerとW&Bを使用した感情分析器のデプロイに関する[チュートリアル](https://wandb.ai/authors/sagemaker/reports/Deploy-Sentiment-Analyzer-Using-SageMaker-and-W-B--VmlldzoxODA1ODE)もご覧いただけます。
:::
:::caution

W&B スイープエージェントは、SageMaker のインテグレーションが無効になっていない場合、SageMaker ジョブで期待どおりの振る舞いをしません。SageMaker のインテグレーションを runs で無効にするには、`wandb.init` の呼び出しを以下のように変更してください。

```
wandb.init(..., settings=wandb.Settings(sagemaker_disable=True))
```

:::