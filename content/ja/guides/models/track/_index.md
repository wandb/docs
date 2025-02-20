---
title: Experiments
description: W&B で機械学習実験をトラッキングします。
cascade:
- url: guides/track/:filename
menu:
  default:
    identifier: ja-guides-models-track-_index
    parent: w-b-models
url: guides/track
weight: 1
---

{{< cta-button productLink="https://wandb.ai/stacey/deep-drive/workspace?workspace=user-lavanyashukla" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases.ipynb" >}}

数行のコードで機械学習実験を管理することができます。その後、結果を[インタラクティブダッシュボード]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}})で確認したり、[Public API]({{< relref path="/ref/python/public-api/" lang="ja" >}})を使用して、プログラムからのアクセスのためにデータをPythonにエクスポートすることができます。

[PyTorch]({{< relref path="/guides/integrations/pytorch.md" lang="ja" >}})、[Keras]({{< relref path="/guides/integrations/keras.md" lang="ja" >}})、[Scikit]({{< relref path="/guides/integrations/scikit.md" lang="ja" >}})などの人気のあるフレームワークを使用する場合は、W&Bインテグレーションを利用してください。[インテグレーションガイド]({{< relref path="/guides/integrations/" lang="ja" >}})で、すべてのインテグレーションのリストやW&Bをコードに追加する方法についての情報を見ることができます。

{{< img src="/images/experiments/experiments_landing_page.png" alt="" >}}

上の画像は、複数の[runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}})間でメトリクスを表示および比較できるダッシュボードの例を示しています。

## 仕組み

数行のコードで機械学習実験を管理します:
1. [W&B run]({{< relref path="/guides/models/track/runs/" lang="ja" >}})を作成します。
2. 学習率やモデルタイプなどのハイパーパラメーターの辞書を設定に保存します（[`wandb.config`]({{< relref path="/guides/models/track/config.md" lang="ja" >}})）。
3. トレーニングループで時間経過とともにメトリクスをログします（例：精度や損失）（[`wandb.log()`]({{< relref path="/guides/models/track/log/" lang="ja" >}})）。
4. run の出力を保存し、モデルの重みや予測のテーブルなどを保存します。

以下の疑似コードは、一般的なW&B実験管理ワークフローを示しています：

```python showLineNumbers
# 1. W&B Run を開始します
wandb.init(entity="", project="my-project-name")

# 2. モデルの入力とハイパーパラメーターを保存します
wandb.config.learning_rate = 0.01

# モデルとデータをインポートします
model, dataloader = get_model(), get_data()

# モデルトレーニングのコードがここに入ります

# 3. 時間をかけてメトリクスをログし、パフォーマンスを視覚化します
wandb.log({"loss": loss})

# 4. W&B にアーティファクトをログします
wandb.log_artifact(model)
```

## 開始方法

ユースケースに応じて、W&B Experiments を開始するための次のリソースを探索してください：

* [W&B クイックスタート]({{< relref path="/guides/quickstart.md" lang="ja" >}})を読んで、W&B Python SDK コマンドを使用してデータセットアーティファクトを作成、管理、使用するためのステップバイステップの概要をつかむ。
* このチャプターを探索して以下を学びます：
  * 実験を作成する方法
  * 実験を設定する方法
  * 実験からデータをログする方法
  * 実験の結果を見る方法
* [W&B API リファレンスガイド]({{< relref path="/ref/" lang="ja" >}})内の[W&B Python ライブラリ]({{< relref path="/ref/python/" lang="ja" >}})を探索します。