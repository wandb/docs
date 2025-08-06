---
title: 実験管理
description: W&B を使って機械学習実験をトラッキングしましょう。
menu:
  default:
    identifier: experiments
    parent: w-b-models
url: guides/track
weight: 1
cascade:
- url: guides/track/:filename
---

{{< cta-button productLink="https://wandb.ai/stacey/deep-drive/workspace?workspace=user-lavanyashukla" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases.ipynb" >}}

数行のコードで機械学習実験をトラッキングできます。実験結果は[インタラクティブなダッシュボード]({{< relref "/guides/models/track/workspaces.md" >}})で確認でき、また [Public API]({{< relref "/ref/python/public-api/index.md" >}}) を使って Python でデータへプログラム的にアクセスできます。

[PyTorch]({{< relref "/guides/integrations/pytorch.md" >}})、[Keras]({{< relref "/guides/integrations/keras.md" >}})、[Scikit]({{< relref "/guides/integrations/scikit.md" >}}) などの人気フレームワークを利用している場合は W&B インテグレーションが便利です。全インテグレーションとコードへの組み込み方法は [Integration guides]({{< relref "/guides/integrations/" >}}) をご覧ください。

{{< img src="/images/experiments/experiments_landing_page.png" alt="Experiments dashboard" >}}

上の画像は、複数の [runs]({{< relref "/guides/models/track/runs/" >}}) のメトリクスを確認・比較できるダッシュボード例です。

## 仕組み

機械学習実験は数行のコードでトラッキングできます：

1. [W&B Run]({{< relref "/guides/models/track/runs/" >}})を作成します。
2. 学習率やモデルタイプなどのハイパーパラメータを、設定用の辞書型で保存します（[`wandb.Run.config`]({{< relref "./config.md" >}})）。
3. トレーニングループ内で、精度や損失などのメトリクスを [`wandb.Run.log()`]({{< relref "/guides/models/track/log/" >}}) で継続的に記録します。
4. モデルの重みや予測結果のテーブルなど、run の出力を保存します。

以下のコードは、よく使われる W&B 実験管理ワークフローの例です：

```python
# run を開始します。
#
# このブロックを抜けるときに、記録したデータのアップロードが完了するまで待機します。
# 例外が発生した場合、run は失敗としてマークされます。
with wandb.init(entity="", project="my-project-name") as run:
  # モデル入力やハイパーパラメータを保存
  run.config.learning_rate = 0.01

  # 実験コードを実行
  for epoch in range(num_epochs):
    # トレーニングを実行...

    # モデル性能の可視化のため、メトリクスを継続的にログします
    run.log({"loss": loss})

  # モデルの出力を artifact としてアップロード
  run.log_artifact(model)
```

## はじめに

ユースケースに応じて、W&B Experiments を始めるための以下のリソースを活用してください：

* [W&B クイックスタート]({{< relref "/guides/quickstart.md" >}}) では、W&B Python SDK のコマンドによるデータセットアーティファクトの作成・トラッキング・利用方法の手順がまとめられています。
* このチャプターを通して以下が学べます：
  * 実験の作成
  * 実験の設定
  * 実験データのログ
  * 実験結果の閲覧
* [W&B API リファレンスガイド]({{< relref "/ref/" >}})の中にある [W&B Python Library]({{< relref "/ref/python/index.md" >}}) もご覧ください。

## ベストプラクティスとヒント

実験管理やログ記録のベストプラクティスとヒントについては、[Best Practices: Experiments and Logging](https://wandb.ai/wandb/pytorch-lightning-e2e/reports/W-B-Best-Practices-Guide--VmlldzozNTU1ODY1#w&b-experiments-and-logging) をご参照ください。