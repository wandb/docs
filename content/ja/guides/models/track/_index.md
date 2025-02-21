---
title: Experiments
description: W&B で機械学習 の 実験管理 を行いましょう。
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

数行のコードで 機械学習 の 実験 を追跡します。次に、[インタラクティブ ダッシュボード]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) で 結果 を確認したり、[Public API]({{< relref path="/ref/python/public-api/" lang="ja" >}}) を 使用して プログラム で アクセス できるように データ を Python にエクスポートしたりできます。

[PyTorch]({{< relref path="/guides/integrations/pytorch.md" lang="ja" >}}), [Keras]({{< relref path="/guides/integrations/keras.md" lang="ja" >}}), or [Scikit]({{< relref path="/guides/integrations/scikit.md" lang="ja" >}}) などの一般的な フレームワーク を使用する場合は、W&B インテグレーション を活用してください。インテグレーション の完全なリストと、W&B を コード に追加する方法については、[インテグレーション ガイド]({{< relref path="/guides/integrations/" lang="ja" >}}) を参照してください。

{{< img src="/images/experiments/experiments_landing_page.png" alt="" >}}

上の図は、複数の [runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) にわたって メトリクス を表示および比較できる ダッシュボード の例を示しています。

## 仕組み

数行のコードで 機械学習 の 実験 を追跡します。
1. [W&B run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を作成します。
2. 学習率や モデル タイプなどの ハイパーパラメーター の 辞書 を 設定 ([`wandb.config`]({{< relref path="/guides/models/track/config.md" lang="ja" >}})) に保存します。
3. トレーニング ループで、精度や 損失 などの メトリクス ([`wandb.log()`]({{< relref path="/guides/models/track/log/" lang="ja" >}})) を 経時的に ログ に記録します。
4. モデル の 重み や 予測 の テーブル など、run の 出力 を保存します。

次の 疑似コード は、一般的な W&B 実験 管理 の ワークフロー を示しています。

```python showLineNumbers
# 1. Start a W&B Run
wandb.init(entity="", project="my-project-name")

# 2. Save mode inputs and hyperparameters
wandb.config.learning_rate = 0.01

# Import model and data
model, dataloader = get_model(), get_data()

# Model training code goes here

# 3. Log metrics over time to visualize performance
wandb.log({"loss": loss})

# 4. Log an artifact to W&B
wandb.log_artifact(model)
```

## 開始方法

ユースケース に応じて、次の リソース を調べて W&B Experiments を開始してください。

* データセット Artifact を作成、追跡、および使用するために使用できる W&B Python SDK コマンド の段階的な概要については、[W&B クイックスタート]({{< relref path="/guides/quickstart.md" lang="ja" >}}) をお読みください。
* この チャプター を調べて、次の方法を学びます。
  * 実験 を作成する
  * 実験 を 設定 する
  * 実験 から データ を ログ に記録する
  * 実験 の 結果 を表示する
* [W&B API Reference Guide]({{< relref path="/ref/" lang="ja" >}}) 内の [W&B Python Library]({{< relref path="/ref/python/" lang="ja" >}}) を調べます。
