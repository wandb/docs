---
title: Experiments
description: W&B で 機械学習 の 実験 を トラックします。
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

数行のコードで 機械学習 の 実験 を追跡します。次に、[インタラクティブ ダッシュボード]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}})で 結果 を確認するか、[Public API]({{< relref path="/ref/python/public-api/" lang="ja" >}})を使用して、プログラムで アクセス できるように データ を Python にエクスポートできます。

[PyTorch]({{< relref path="/guides/integrations/pytorch.md" lang="ja" >}}), [Keras]({{< relref path="/guides/integrations/keras.md" lang="ja" >}}), or [Scikit]({{< relref path="/guides/integrations/scikit.md" lang="ja" >}})のような一般的な フレームワーク を使用する場合は、W&B インテグレーション を活用してください。インテグレーション の完全なリストと、W&B を コード に追加する方法については、[インテグレーション ガイド]({{< relref path="/guides/integrations/" lang="ja" >}})を参照してください。

{{< img src="/images/experiments/experiments_landing_page.png" alt="" >}}

上の図は、複数の [runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}})にわたって メトリクス を表示および比較できる ダッシュボード の例を示しています。

## 仕組み

数行の コード で 機械学習 の 実験 を追跡します。
1. [W&B run]({{< relref path="/guides/models/track/runs/" lang="ja" >}})を作成します。
2. 学習率や モデル タイプなどの ハイパーパラメーター の 辞書 を 設定 ([`run.config`]({{< relref path="./config.md" lang="ja" >}}))に保存します。
3. トレーニング ループで、精度や 損失 などの メトリクス ([`run.log()`]({{< relref path="/guides/models/track/log/" lang="ja" >}}))を ログ に記録します。
4. モデル の 重みや 予測 の テーブル など、run の 出力 を保存します。

次の コード は、一般的な W&B の 実験 管理 ワークフロー を示しています。

```python
# Start a run.
#
# When this block exits, it waits for logged data to finish uploading.
# If an exception is raised, the run is marked failed.
with wandb.init(entity="", project="my-project-name") as run:
  # Save mode inputs and hyperparameters.
  run.config.learning_rate = 0.01

  # Run your experiment code.
  for epoch in range(num_epochs):
    # Do some training...

    # Log metrics over time to visualize model performance.
    run.log({"loss": loss})

  # Upload model outputs as artifacts.
  run.log_artifact(model)
```

## はじめに

ユースケース に応じて、次の リソース を調べて W&B Experiments を開始してください。

* データセット Artifact を作成、追跡、および使用するために使用できる W&B Python SDK コマンド のステップごとの 概要 については、[W&B クイックスタート]({{< relref path="/guides/quickstart.md" lang="ja" >}})をお読みください。
* この チャプター を調べて、次の方法を学びます。
  * 実験 を作成する
  * 実験 を 設定 する
  * 実験 から データ を ログ に記録する
  * 実験 の 結果 を表示する
* [W&B API Reference Guide]({{< relref path="/ref/" lang="ja" >}})内の [W&B Python Library]({{< relref path="/ref/python/" lang="ja" >}})を調べます。

## ベストプラクティス と ヒント

実験 と ログ の ベストプラクティス と ヒント については、[Best Practices: Experiments and Logging](https://wandb.ai/wandb/pytorch-lightning-e2e/reports/W-B-Best-Practices-Guide--VmlldzozNTU1ODY1#w&b-experiments-and-logging)を参照してください。
