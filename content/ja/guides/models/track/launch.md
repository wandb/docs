---
title: Create an experiment
description: W&B の 実験 を作成します。
menu:
  default:
    identifier: ja-guides-models-track-launch
    parent: experiments
weight: 1
---

W&B Python SDK を使用して、 機械学習 の 実験 を追跡します。次に、インタラクティブな ダッシュボード で 結果 を確認するか、[W&B Public API]({{< relref path="/ref/python/public-api/" lang="ja" >}}) でプログラムから アクセス できるように データ を Python にエクスポートできます。

この ガイド では、W&B の構成要素を使用して W&B の 実験 を作成する方法について説明します。

## W&B の 実験 を作成する方法

W&B の 実験 は、次の 4 つの ステップ で作成します。

1. [W&B Run の初期化]({{< relref path="#initialize-a-wb-run" lang="ja" >}})
2. [ハイパーパラメーター の 辞書 をキャプチャ]({{< relref path="#capture-a-dictionary-of-hyperparameters" lang="ja" >}})
3. [トレーニング ループ 内で メトリクス を ログ 記録]({{< relref path="#log-metrics-inside-your-training-loop" lang="ja" >}})
4. [W&B に アーティファクト を ログ 記録]({{< relref path="#log-an-artifact-to-wb" lang="ja" >}})

### W&B run の初期化
[`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ja" >}}) を使用して、W&B Run を作成します。

次の コードスニペット は、この run を識別するために、説明が `“My first experiment”` である `“cat-classification”` という名前の W&B の プロジェクト で run を作成します。タグ `“baseline”` および `“paper1”` は、この run が将来の論文出版を目的とした ベースライン 実験 であることを思い出させるために含まれています。

```python
import wandb

with wandb.init(
    project="cat-classification",
    notes="My first experiment",
    tags=["baseline", "paper1"],
) as run:
    ...
```

`wandb.init()` は、[Run]({{< relref path="/ref/python/run.md" lang="ja" >}}) オブジェクト を返します。

{{% alert %}}
注意：run は、`wandb.init()` を呼び出すときに プロジェクト が既に存在する場合、既存の プロジェクト に追加されます。たとえば、`“cat-classification”` という プロジェクト が既にある場合、その プロジェクト は引き続き存在し、削除されません。代わりに、新しい run がその プロジェクト に追加されます。
{{% /alert %}}

### ハイパーパラメーター の 辞書 をキャプチャ
学習率や モデル タイプなどの ハイパーパラメーター の 辞書 を保存します。config でキャプチャする モデル の 設定 は、 後で 結果 を整理してクエリするのに役立ちます。

```python
with wandb.init(
    ...,
    config={"epochs": 100, "learning_rate": 0.001, "batch_size": 128},
) as run:
    ...
```

実験 の構成方法の詳細については、[実験 の構成]({{< relref path="./config.md" lang="ja" >}}) を参照してください。

### トレーニング ループ 内で メトリクス を ログ 記録
[`run.log()`]({{< relref path="/ref/python/log.md" lang="ja" >}}) を呼び出して、精度や損失など、各トレーニング ステップ に関する メトリクス を ログ 記録します。

```python
model, dataloader = get_model(), get_data()

for epoch in range(run.config.epochs):
    for batch in dataloader:
        loss, accuracy = model.training_step()
        run.log({"accuracy": accuracy, "loss": loss})
```

W&B で ログ 記録できるさまざまな データ タイプ の詳細については、[実験 中の データ の ログ 記録]({{< relref path="/guides/models/track/log/" lang="ja" >}}) を参照してください。

### W&B に アーティファクト を ログ 記録
オプションで、W&B Artifact を ログ 記録します。Artifacts を使用すると、 データセット と モデル を簡単に バージョン管理 できます。
```python
# You can save any file or even a directory. In this example, we pretend
# the model has a save() method that outputs an ONNX file.
model.save("path_to_model.onnx")
run.log_artifact("path_to_model.onnx", name="trained-model", type="model")
```
[Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) または [Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) での モデル の バージョン管理 について詳しく学びましょう。

### まとめ
上記の コードスニペット を使用した完全な スクリプト は、以下にあります。
```python
import wandb

with wandb.init(
    project="cat-classification",
    notes="",
    tags=["baseline", "paper1"],
    # Record the run's hyperparameters.
    config={"epochs": 100, "learning_rate": 0.001, "batch_size": 128},
) as run:
    # Set up model and data.
    model, dataloader = get_model(), get_data()

    # Run your training while logging metrics to visualize model performance.
    for epoch in range(run.config["epochs"]):
        for batch in dataloader:
            loss, accuracy = model.training_step()
            run.log({"accuracy": accuracy, "loss": loss})

    # Upload the trained model as an artifact.
    model.save("path_to_model.onnx")
    run.log_artifact("path_to_model.onnx", name="trained-model", type="model")
```

## 次の ステップ ： 実験 を視覚化する
W&B ダッシュボード を、 機械学習 モデル からの 結果 を整理して視覚化するための一元的な場所として使用します。数回クリックするだけで、[パラレル座標図]({{< relref path="/guides/models/app/features/panels/parallel-coordinates.md" lang="ja" >}})、[パラメータ の重要性分析]({{< relref path="/guides/models/app/features/panels/parameter-importance.md" lang="ja" >}}) などの豊富な インタラクティブ な グラフ を構築できます。[その他]({{< relref path="/guides/models/app/features/panels/" lang="ja" >}}) もあります。

{{< img src="/images/sweeps/quickstart_dashboard_example.png" alt="Quickstart Sweeps Dashboard example" >}}

実験 と特定の run の表示方法の詳細については、[実験 からの 結果 の視覚化]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) を参照してください。

## ベストプラクティス
以下は、 実験 を作成する際に考慮すべき推奨 ガイドライン です。

1.  **run を完了させる**: `wandb.init()` を `with` ステートメント で使用して、 コード が完了するか 例外 が発生したときに、run が自動的に完了としてマークされるようにします。
    * Jupyter ノートブック では、Run オブジェクト を自分で管理する方が便利な場合があります。この場合、Run オブジェクト で `finish()` を明示的に呼び出して、完了としてマークできます。

        ```python
        # In a notebook cell:
        run = wandb.init()

        # In a different cell:
        run.finish()
        ```
2.  **Config**: ハイパーパラメーター 、 architecture 、 データセット 、 モデル を再現するために使用したいその他のものを追跡します。これらは列に表示されます。 config 列を使用して、アプリ で run を動的にグループ化、並べ替え、フィルター処理します。
3.  **Project**: プロジェクト は、一緒に比較できる 実験 の セット です。各 プロジェクト には専用の ダッシュボード ページがあり、さまざまな モデル バージョン を比較するために、さまざまな run のグループを簡単にオン/オフにできます。
4.  **Notes**: スクリプト から直接クイック コミット メッセージ を設定します。W&B App の run の 概要 セクション で ノート を編集して アクセス します。
5.  **Tags**: ベースライン run とお気に入りの run を識別します。タグ を使用して run をフィルター処理できます。W&B App の プロジェクト の ダッシュボード の 概要 セクション で、 後で タグ を編集できます。
6.  **複数の run セット を作成して 実験 を比較する**: 実験 を比較するときは、 メトリクス を簡単に比較できるように複数の run セット を作成します。同じ グラフ または グラフ のグループで run セット のオン/オフを切り替えることができます。

次の コードスニペット は、上記の ベストプラクティス を使用して W&B の 実験 を定義する方法を示しています。

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

W&B の 実験 を定義する際に使用できる パラメータ の詳細については、[API Reference Guide]({{< relref path="/ref/python/" lang="ja" >}}) の [`wandb.init`]({{< relref path="/ref/python/init.md" lang="ja" >}}) API ドキュメント を参照してください。
