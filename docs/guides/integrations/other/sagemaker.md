---
description: W&B を Amazon SageMaker と統合する方法
slug: /guides/integrations/sagemaker
displayed_sidebar: default
---


# SageMaker

## SageMaker インテグレーション

W&B は [Amazon SageMaker](https://aws.amazon.com/sagemaker/) とインテグレートされており、ハイパーパラメーターの自動読み取り、分散 run のグループ化、チェックポイントからの run の再開が可能となります。

### 認証

W&B はトレーニングスクリプトの相対パスにある `secrets.env` というファイルを探し、`wandb.init()` が呼び出されたときにそれらを環境にロードします。実験を開始するスクリプトで `wandb.sagemaker_auth(path="source_dir")` を呼び出すことで `secrets.env` ファイルを生成できます。このファイルを `.gitignore` に追加することを忘れないでください！

### 既存のエスティメーター

SageMaker の事前設定されたエスティメーターの1つを使用している場合、wandb を含む `requirements.txt` をソースディレクトリーに追加する必要があります。

```
wandb
```

Python 2 を実行しているエスティメーターを使用している場合は、wandb をインストールする前に [wheel](https://pythonwheels.com) から直接 psutil をインストールする必要があります:

```
https://wheels.galaxyproject.org/packages/psutil-5.4.8-cp27-cp27mu-manylinux1_x86_64.whl
wandb
```

:::情報
完全な例は [GitHub](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cifar10-sagemaker) で利用可能です。また、[ブログ](https://wandb.ai/site/articles/running-sweeps-with-sagemaker) でも詳細を読むことができます。\
SageMaker と W&B を使用して感情分析器をデプロイする方法についての [チュートリアル](https://wandb.ai/authors/sagemaker/reports/Deploy-Sentiment-Analyzer-Using-SageMaker-and-W-B--VmlldzoxODA1ODE) もご覧になれます。
:::

:::注意
W&B sweep agent は、SageMaker ジョブでは SageMaker インテグレーションが無効化されていない限り、期待どおりに動作しません。次のように `wandb.init` の呼び出しを変更することで、run で SageMaker インテグレーションを無効化できます:

```
wandb.init(..., settings=wandb.Settings(sagemaker_disable=True))
```
:::
