---
title: 実験を作成する
description: W&B Experiment を作成する
menu:
  default:
    identifier: ja-create-an-experiment
    parent: experiments
weight: 1
---

W&B Python SDK を使って機械学習実験をトラッキングしましょう。トラッキングした 実験結果は、インタラクティブなダッシュボードでレビューできるほか、[W&B Public API]({{< relref path="/ref/python/public-api/" lang="ja" >}}) を使ってプログラムから Python でデータをエクスポートして アクセスすることもできます。

このガイドでは、W&B のビルディングブロックを使って W&B Experiment を作成する方法について説明します。

## W&B Experiment を作成する方法

W&B Experiment は、次の4ステップで作成できます。

1. [W&B Run を初期化する]({{< relref path="#initialize-a-wb-run" lang="ja" >}})
2. [ハイパーパラメーターの辞書を保存する]({{< relref path="#capture-a-dictionary-of-hyperparameters" lang="ja" >}})
3. [トレーニングループ内でメトリクスをログする]({{< relref path="#log-metrics-inside-your-training-loop" lang="ja" >}})
4. [W&B に Artifact をログする]({{< relref path="#log-an-artifact-to-wb" lang="ja" >}})

### W&B run を初期化する
[`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init" lang="ja" >}}) を使って W&B Run を作成します。

以下のスニペットは、`cat-classification` という名前の W&B Project で run を作成し、`My first experiment` という説明文を追加しています。この run を特定しやすくするため、`baseline` と `paper1` というタグを付け、「この run は将来発表予定の論文用のベースライン実験である」という意味を持たせています。

```python
import wandb

with wandb.init(
    project="cat-classification",
    notes="My first experiment",
    tags=["baseline", "paper1"],
) as run:
    ...
```

`wandb.init()` は [Run]({{< relref path="/ref/python/sdk/classes/run" lang="ja" >}}) オブジェクトを返します。

{{% alert %}}
注意: `wandb.init()` を呼び出したとき、その Project が既に存在していれば Run は既存の Project に追加されます。たとえば `cat-classification` というプロジェクトが既にある場合、その Project はそのまま残り、新たに Run だけが追加されます。
{{% /alert %}}

### ハイパーパラメーターの辞書を保存する
learning rate や model type など、ハイパーパラメーターの辞書を保存しましょう。config で保存した モデル設定は、後から 結果を整理したり検索したりするときに役立ちます。

```python
with wandb.init(
    ...,
    config={"epochs": 100, "learning_rate": 0.001, "batch_size": 128},
) as run:
    ...
```

実験の設定方法について詳しくは [Configure Experiments]({{< relref path="./config.md" lang="ja" >}}) をご覧ください。

### トレーニングループ内でメトリクスをログする
[`run.log()`]({{< relref path="/ref/python/sdk/classes/run/#method-runlog" lang="ja" >}}) を呼び出すと、トレーニングステップごとの accuracy や loss などのメトリクスを記録できます。

```python
model, dataloader = get_model(), get_data()

for epoch in range(run.config.epochs):
    for batch in dataloader:
        loss, accuracy = model.training_step()
        run.log({"accuracy": accuracy, "loss": loss})
```

W&B で様々なデータタイプをログする方法については [Log Data During Experiments]({{< relref path="/guides/models/track/log/" lang="ja" >}}) をご覧ください。

### W&B に Artifact をログする
必要に応じて W&B Artifact をログしましょう。Artifact はデータセットやモデルのバージョン管理を簡単にします。

```python
# どんなファイルでも、ディレクトリでも保存可能です。ここでは例として
# モデルが save() メソッドを持ち、ONNX ファイルを出力するものとします。
model.save("path_to_model.onnx")
run.log_artifact("path_to_model.onnx", name="trained-model", type="model")
```
[Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) の詳細や、バージョン管理されたモデルについては [Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) もご覧ください。

### すべてをまとめる
上記のコードスニペットをすべて組み合わせた全体のスクリプト例は以下の通りです。

```python
import wandb

with wandb.init(
    project="cat-classification",
    notes="",
    tags=["baseline", "paper1"],
    # run のハイパーパラメーターを記録
    config={"epochs": 100, "learning_rate": 0.001, "batch_size": 128},
) as run:
    # モデルとデータセットを準備
    model, dataloader = get_model(), get_data()

    # モデルのパフォーマンスを可視化できるよう、メトリクスをログしながらトレーニングを実行
    for epoch in range(run.config["epochs"]):
        for batch in dataloader:
            loss, accuracy = model.training_step()
            run.log({"accuracy": accuracy, "loss": loss})

    # 訓練済みモデルを Artifact としてアップロード
    model.save("path_to_model.onnx")
    run.log_artifact("path_to_model.onnx", name="trained-model", type="model")
```

## 次のステップ: experiment を可視化する
W&B ダッシュボードは、 機械学習モデルの結果を一元管理・可視化するための中心的な場所です。いくつかのクリックだけで、[parallel coordinates プロット]({{< relref path="/guides/models/app/features/panels/parallel-coordinates.md" lang="ja" >}})、[パラメータの重要度分析]({{< relref path="/guides/models/app/features/panels/parameter-importance.md" lang="ja" >}})、[その他のグラフ種類]({{< relref path="/guides/models/app/features/panels/" lang="ja" >}}) など、豊富なインタラクティブなグラフを構築できます。

{{< img src="/images/sweeps/quickstart_dashboard_example.png" alt="Quickstart Sweeps Dashboard example" >}}

experiment および特定の run の可視化方法について詳しくは [Visualize results from experiments]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) をご参照ください。

## ベストプラクティス
Experiment を作成する際に参考になる推奨ガイドラインをいくつか紹介します。

1. **run を確実に終了する**: `wandb.init()` を `with` 文で使えば、コードの実行完了時や例外発生時に run を自動で終了できます。
    * Jupyter ノートブックの場合、Run オブジェクトを自分で管理する方が便利な場合があります。この場合は、Run オブジェクトで `finish()` を明示的に呼び出してください。

        ```python
        # ノートブックセルでは:
        run = wandb.init()

        # 別のセルで:
        run.finish()
        ```
2. **Config を活用する**: ハイパーパラメーター、アーキテクチャー、データセットなど、 モデル再現に必要な情報を記録しましょう。これらはカラム表示され、config カラムで run を動的にグループ・ソート・フィルターできます。
3. **Project**: Project は比較したい experiment をまとめる単位です。各 Project には専用のダッシュボードページが用意され、異なるグループの run を切り替えて様々なモデルバージョンを簡単に比較できます。
4. **Notes**: スクリプトから直接短いコミットメッセージを設定しましょう。W&B App の run の Overview セクションで notes を編集・閲覧できます。
5. **Tags**: ベースライン run やお気に入りの run をタグで識別しましょう。タグで run をフィルタリングできます。タグはプロジェクトのダッシュボード Overview セクションからあとで編集も可能です。
6. **複数の run セットを作成して experiment を比較する**: experiment を比較するときは、複数の run セットを作成するとメトリクス比較が容易になります。同じグラフやグループで run セットをオン・オフして比較できます。

以下のコードスニペットは、上記のベストプラクティスを踏まえて W&B Experiment を定義する方法の一例です。

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

W&B Experiment を定義するときに指定できるパラメータの詳細については、[API Reference Guide]({{< relref path="/ref/python/" lang="ja" >}}) の [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init" lang="ja" >}}) API ドキュメントをご覧ください。