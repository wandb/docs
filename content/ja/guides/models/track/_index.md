---
title: 実験管理
description: W&B で機械学習実験を追跡する。
cascade:
- url: /ja/guides/track/:filename
menu:
  default:
    identifier: ja-guides-models-track-_index
    parent: w-b-models
url: /ja/guides/track
weight: 1
---

{{< cta-button productLink="https://wandb.ai/stacey/deep-drive/workspace?workspace=user-lavanyashukla" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases.ipynb" >}}

数行のコードで機械学習実験を追跡します。その後、[インタラクティブなダッシュボード]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}})で結果をレビューしたり、[Public API]({{< relref path="/ref/python/public-api/" lang="ja" >}})を使用してプログラムからアクセスできるようにPythonにデータをエクスポートすることができます。

人気のあるフレームワークを使用している場合は、W&Bのインテグレーションを活用してください。[PyTorch]({{< relref path="/guides/integrations/pytorch.md" lang="ja" >}})、[Keras]({{< relref path="/guides/integrations/keras.md" lang="ja" >}})、または[Scikit]({{< relref path="/guides/integrations/scikit.md" lang="ja" >}})のようなフレームワークがあります。インテグレーションの完全なリストや、W&Bをコードに追加する方法については、[インテグレーションガイド]({{< relref path="/guides/integrations/" lang="ja" >}})をご覧ください。

{{< img src="/images/experiments/experiments_landing_page.png" alt="" >}}

上の画像は、複数の[Runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}})でメトリクスを確認および比較できるダッシュボードの例を示しています。

## 仕組み

数行のコードで機械学習実験を追跡します:
1. [W&B Run]({{< relref path="/guides/models/track/runs/" lang="ja" >}})を作成します。
2. 学習率やモデルタイプなどのハイパーパラメーターを辞書として設定（[`run.config`]({{< relref path="./config.md" lang="ja" >}})）に保存します。
3. トレーニングループ中に正確性や損失などのメトリクスをログ（[`run.log()`]({{< relref path="/guides/models/track/log/" lang="ja" >}})）します。
4. モデルの重みや予測のテーブルのようなRunの出力を保存します。

以下のコードは、一般的なW&B実験管理ワークフローを示しています:

```python
# Run を開始します。
#
# このブロックから出ると、ログデータのアップロードが完了するのを待ちます。
# 例外が発生した場合、Run は失敗としてマークされます。
with wandb.init(entity="", project="my-project-name") as run:
  # モード入力とハイパーパラメーターを保存します。
  run.config.learning_rate = 0.01

  # 実験コードを実行します。
  for epoch in range(num_epochs):
    # トレーニングをします...

    # モデルのパフォーマンスを可視化するためのメトリクスを時間と共にログします。
    run.log({"loss": loss})

  # モデルの出力をアーティファクトとしてアップロードします。
  run.log_artifact(model)
```

## 始めましょう

あなたのユースケースに応じて、W&B Experimentsの開始に役立つ次のリソースを探索してください:

* [W&Bクイックスタート]({{< relref path="/guides/quickstart.md" lang="ja" >}})を読んで、W&B Python SDKコマンドを使用してデータセットアーティファクトを作成、追跡、および利用するためのステップバイステップの概要を確認してください。
* このチャプターを探索して、以下を学びましょう:
  * 実験を作成する
  * 実験を設定する
  * 実験からデータをログする
  * 実験から結果を確認する
* [W&B APIリファレンスガイド]({{< relref path="/ref/" lang="ja" >}})内の[W&B Pythonライブラリ]({{< relref path="/ref/python/" lang="ja" >}})を探索してください。

## ベストプラクティスとヒント

実験とログのベストプラクティスとヒントについては、[ベストプラクティス： 実験とログ](https://wandb.ai/wandb/pytorch-lightning-e2e/reports/W-B-Best-Practices-Guide--VmlldzozNTU1ODY1#w&b-experiments-and-logging)をご覧ください。