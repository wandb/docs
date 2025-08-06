---
title: 実験を作成
description: W&B Experiment を作成する
menu:
  default:
    identifier: launch
    parent: experiments
weight: 1
---

W&B Python SDK を使って機械学習実験をトラッキングしましょう。実験結果はインタラクティブなダッシュボードで確認でき、[W&B Public API]({{< relref "/ref/python/public-api/" >}}) を使って Python 上でプログラム的にデータを取得することもできます。

このガイドでは、W&B の基本機能を活用して W&B Experiment を作成する方法を説明します。

## W&B Experiment 作成手順

W&B Experiment は以下の4ステップで作成します。

1. [W&B Run を初期化する]({{< relref "#initialize-a-wb-run" >}})
2. [ハイパーパラメータの辞書を保存する]({{< relref "#capture-a-dictionary-of-hyperparameters" >}})
3. [トレーニングループ内でメトリクスをログする]({{< relref "#log-metrics-inside-your-training-loop" >}})
4. [Artifact を W&B にログする]({{< relref "#log-an-artifact-to-wb" >}})

### W&B Run を初期化する

[`wandb.init()`]({{< relref "/ref/python/sdk/functions/init" >}}) を使って W&B Run を作成します。

次のスニペットでは、`"cat-classification"` という名前の W&B Project 内に run を作成しています。run の説明は `"My first experiment"` で、run の識別に役立ちます。また `"baseline"` と `"paper1"` のタグをつけて、この run が論文用のベースライン実験であることを示しています。

```python
import wandb

with wandb.init(
    project="cat-classification",
    notes="My first experiment",
    tags=["baseline", "paper1"],
) as run:
    ...
```

`wandb.init()` は [Run]({{< relref "/ref/python/sdk/classes/run" >}}) オブジェクトを返します。

{{% alert %}}
注意: `wandb.init()` 実行時、既存の Project がある場合はその Project に Run が追加されます。たとえば、すでに `"cat-classification"` という Project がある場合、その Project は削除されず、そのまま新しい run が追加されていきます。
{{% /alert %}}

### ハイパーパラメータの辞書を保存する

学習率やモデル種別などのハイパーパラメータの辞書を保存します。config で記録したモデルの設定は、後の結果整理や検索にも役立ちます。

```python
with wandb.init(
    ...,
    config={"epochs": 100, "learning_rate": 0.001, "batch_size": 128},
) as run:
    ...
```

実験の設定方法について詳細は [Configure Experiments]({{< relref "./config.md" >}}) をご覧ください。

### トレーニングループ内でメトリクスをログする

[`run.log()`]({{< relref "/ref/python/sdk/classes/run/#method-runlog" >}}) を使い、各トレーニングステップごとに accuracy や loss などのメトリクスをログします。

```python
model, dataloader = get_model(), get_data()

for epoch in range(run.config.epochs):
    for batch in dataloader:
        loss, accuracy = model.training_step()
        run.log({"accuracy": accuracy, "loss": loss})
```

W&B でログできるデータタイプの詳細は [Log Data During Experiments]({{< relref "/guides/models/track/log/" >}}) を参照してください。

### Artifact を W&B にログする

必要に応じて W&B Artifact を保存します。Artifact を使うとデータセットやモデルのバージョン管理が容易になります。

```python
# 任意のファイルやディレクトリを保存できます。この例では
# model の save() メソッドが ONNX ファイルを出力すると仮定しています。
model.save("path_to_model.onnx")
run.log_artifact("path_to_model.onnx", name="trained-model", type="model")
```
[Artifacts]({{< relref "/guides/core/artifacts/" >}}) や [Registry]({{< relref "/guides/core/registry/" >}}) でのモデルのバージョン管理について詳細をご覧ください。

### まとめ：全体のスクリプト

前述のコードスニペットをまとめた全体のサンプルは以下の通りです。

```python
import wandb

with wandb.init(
    project="cat-classification",
    notes="",
    tags=["baseline", "paper1"],
    # run のハイパーパラメータを記録します。
    config={"epochs": 100, "learning_rate": 0.001, "batch_size": 128},
) as run:
    # モデルとデータのセットアップ
    model, dataloader = get_model(), get_data()

    # モデル性能を可視化しながらトレーニング。
    for epoch in range(run.config["epochs"]):
        for batch in dataloader:
            loss, accuracy = model.training_step()
            run.log({"accuracy": accuracy, "loss": loss})

    # トレーニング済みモデルを Artifact としてアップロード
    model.save("path_to_model.onnx")
    run.log_artifact("path_to_model.onnx", name="trained-model", type="model")
```

## 次のステップ：Experiment を可視化する

W&B Dashboard を使うと、機械学習モデルの結果整理や可視化を一元化できます。数クリックでリッチなインタラクティブグラフ（[平行座標プロット]({{< relref "/guides/models/app/features/panels/parallel-coordinates.md" >}})、[パラメータの重要度分析]({{< relref "/guides/models/app/features/panels/parameter-importance.md" >}})、[その他グラフ]({{< relref "/guides/models/app/features/panels/" >}}) など）を作成できます。

{{< img src="/images/sweeps/quickstart_dashboard_example.png" alt="Quickstart Sweeps Dashboard example" >}}

実験や特定の Run の閲覧方法については [Visualize results from experiments]({{< relref "/guides/models/track/workspaces.md" >}}) をご参照ください。

## ベストプラクティス

Experiment 作成時におすすめするガイドラインを紹介します。

1. **Run の終了を忘れずに**: `wandb.init()` を `with` 文で使うことで、コードの実行終了や例外発生時に自動的に Run が終了としてマークされます。
    * Jupyter ノートブックの場合は、自分で Run オブジェクトを管理した方が便利な場合があります。その場合は Run オブジェクトの `finish()` を手動で呼んで終了してください。

        ```python
        # ノートブックセル内で
        run = wandb.init()

        # 別のセルで
        run.finish()
        ```
2. **Config**: ハイパーパラメータ、アーキテクチャー、データセットなど、モデルの再現に必要な情報はすべて記録しましょう。これらはカラムとして表示され、アプリ内でグループ化・ソート・フィルタが簡単にできます。
3. **Project**: Project は比較したい実験のまとまりです。Project ごとに専用のダッシュボードページが生成され、異なる Run グループを簡単に切り替えてモデルバージョンを比較できます。
4. **Notes**: スクリプトから素早くコミットメッセージを設定できます。W&B App の Run の Overview セクションでメモを編集・参照可能です。
5. **Tags**: ベースラインやお気に入りの Run の識別にタグを使いましょう。タグで Run をフィルタでき、プロジェクトのダッシュボード Overview セクションから後で編集も可能です。
6. **複数の Run セットを作成して実験比較**: 実験を比較する際は、複数の Run セットを作成し、メトリクスの比較をしやすくしましょう。同じグラフやグラフグループ上でセットごとの表示切り替えが可能です。

次のコードスニペットは、上記ベストプラクティスを反映した W&B Experiment 定義例です。

```python
import wandb

config = {
    "learning_rate": 0.01,
    "momentum": 0.2,
    "architecture": "CNN",
    "dataset_id": "cats-0192",
}

with wandb.init(
    project="detect-cats",
    notes="tweak baseline",
    tags=["baseline", "paper1"],
    config=config,
) as run:
    ...
```

W&B Experiment 定義時に利用できるパラメータについては、[API Reference Guide]({{< relref "/ref/python/" >}}) 内の [`wandb.init()`]({{< relref "/ref/python/sdk/functions/init" >}}) API ドキュメントも参照してください。