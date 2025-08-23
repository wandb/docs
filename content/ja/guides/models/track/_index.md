---
title: 実験管理
description: W&B で機械学習実験をトラッキングしましょう。
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

数行のコードで機械学習実験をトラッキングできます。結果は[インタラクティブなダッシュボード]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}})で確認したり、[Public API]({{< relref path="/ref/python/public-api/index.md" lang="ja" >}})を使って Python へデータをエクスポートしプログラムからもアクセスできます。

PyTorch や [Keras]({{< relref path="/guides/integrations/keras.md" lang="ja" >}})、[Scikit]({{< relref path="/guides/integrations/scikit.md" lang="ja" >}})といった人気のフレームワークをご利用の場合は、W&B Integrations を活用してください。[Integration guides]({{< relref path="/guides/integrations/" lang="ja" >}}) では対応するすべてのインテグレーションやコードへの導入方法をご紹介しています。

{{< img src="/images/experiments/experiments_landing_page.png" alt="Experiments dashboard" >}}

上の画像は、複数の [runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) におけるメトリクスの比較や可視化ができるダッシュボードの一例です。

## 仕組み

数行のコードで機械学習実験をトラッキングできます：
1. [W&B Run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を作成します。
2. 学習率やモデルタイプなどのハイパーパラメーターを辞書として設定に保存します（[`wandb.Run.config`]({{< relref path="./config.md" lang="ja" >}})）。
3. トレーニングループ内で、精度や損失などのメトリクスを [`wandb.Run.log()`]({{< relref path="/guides/models/track/log/" lang="ja" >}}) で記録します。
4. モデルの重みや予測表など、run の成果物を保存します。

以下のコードは、一般的な W&B 実験管理ワークフローの例です：

```python
# Run を開始します。
#
# このブロックを抜けると、記録したデータのアップロードを待機します。
# 例外が発生した場合、run は失敗としてマークされます。
with wandb.init(entity="", project="my-project-name") as run:
  # モデルの入力やハイパーパラメーターを保存します。
  run.config.learning_rate = 0.01

  # 実験コードを実行します。
  for epoch in range(num_epochs):
    # トレーニング処理...

    # モデルのパフォーマンスを可視化するため、メトリクスを記録します。
    run.log({"loss": loss})

  # モデルの成果物を artifacts としてアップロードします。
  run.log_artifact(model)
```

## はじめに

ご自身のユースケースに応じて、W&B Experiments の利用開始には次のリソースをご活用ください：

* [W&B クイックスタート]({{< relref path="/guides/quickstart.md" lang="ja" >}}) では、データセット artifact を作成し、トラッキングし、活用するまでの一連の W&B Python SDK コマンドの流れをご紹介しています。
* このチャプターでは下記について学べます：
  * 実験の作成
  * 実験の設定
  * 実験からのデータのログ
  * 実験結果の閲覧
* [W&B Python Library]({{< relref path="/ref/python/index.md" lang="ja" >}}) や [W&B APIリファレンスガイド]({{< relref path="/ref/" lang="ja" >}}) もぜひご参照ください。

## ベストプラクティスとヒント

実験やログのベストプラクティスやヒントについては、[Best Practices: Experiments and Logging](https://wandb.ai/wandb/pytorch-lightning-e2e/reports/W-B-Best-Practices-Guide--VmlldzozNTU1ODY1#w&b-experiments-and-logging) をご覧ください。