---
title: Experiments
description: W&B で 機械学習の実験を追跡しましょう。
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

数行のコードで機械学習の実験をトラッキングできます。結果は [インタラクティブなダッシュボード]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) で確認するか、[Public API]({{< relref path="/ref/python/public-api/index.md" lang="ja" >}}) を使ってデータを Python にエクスポートし、プログラムから アクセス できます。

PyTorch、Keras、Scikit などの人気フレームワークを使っている場合は W&B Integrations を活用してください。インテグレーションの一覧やコードに W&B を追加する方法は [Integration guides]({{< relref path="/guides/integrations/" lang="ja" >}}) を参照してください。

{{< img src="/images/experiments/experiments_landing_page.png" alt="Experiments ダッシュボード" >}}

上の画像は、複数の [runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) にわたるメトリクスを表示・比較できるダッシュボードの例です。

## 仕組み

数行のコードで機械学習の実験をトラッキングします:
1. [W&B Run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を作成します。
2. 学習率やモデル種別などのハイパーパラメーターの辞書を、設定（[`wandb.Run.config`]({{< relref path="./config.md" lang="ja" >}})）に保存します。
3. 精度や損失などのメトリクスを、トレーニング ループ内で時間経過に沿って（[`wandb.Run.log()`]({{< relref path="/guides/models/track/log/" lang="ja" >}})）記録します。
4. モデルの重みや予測テーブルなど、run の出力を保存します。

以下は、一般的な W&B の実験管理ワークフローを示すコード例です:

```python
# run を開始します。
#
# このブロックを抜けると、記録したデータのアップロード完了を待ちます。
# 例外が発生した場合、その run は失敗としてマークされます。
with wandb.init(entity="", project="my-project-name") as run:
  # 入力やハイパーパラメーターを保存します。
  run.config.learning_rate = 0.01

  # 実験のコードを実行します。
  for epoch in range(num_epochs):
    # ここで学習を行う...

    # モデルの性能を可視化できるよう、メトリクスを時系列でログします。
    run.log({"loss": loss})

  # モデルの出力を Artifacts としてアップロードします。
  run.log_artifact(model)
```

## はじめに

ユースケースに応じて、W&B Experiments を始めるために次のリソースを参照してください:

* データセットのアーティファクトを作成、トラッキング、活用するために使用できる W&B Python SDK のコマンドを段階的に解説した [W&B Quickstart]({{< relref path="/guides/quickstart.md" lang="ja" >}}) を読んでください。
* このチャプターでは次の内容を学べます:
  * 実験を作成する
  * 実験を設定する
  * 実験からデータをログに記録する
  * 実験の結果を確認する
* [W&B API Reference Guide]({{< relref path="/ref/" lang="ja" >}}) 内の [W&B Python Library]({{< relref path="/ref/python/index.md" lang="ja" >}}) も参照してください。

## ベストプラクティスとヒント 

実験とロギングのベストプラクティスやヒントについては、[Best Practices: Experiments and Logging](https://wandb.ai/wandb/pytorch-lightning-e2e/reports/W-B-Best-Practices-Guide--VmlldzozNTU1ODY1#w&b-experiments-and-logging) を参照してください。