---
title: 実験を作成する
description: W&B Experiment を作成する。
menu:
  default:
    identifier: ja-guides-models-track-create-an-experiment
    parent: experiments
weight: 1
---

W&B Python SDK を使って 機械学習 の実験を記録しましょう。結果は インタラクティブな ダッシュボード で確認でき、[W&B Public API]({{< relref path="/ref/python/public-api/" lang="ja" >}}) を使って データ を Python にエクスポートし、プログラムから アクセス することも可能です。

このガイドでは、W&B のビルディングブロックを使って W&B Experiment を作成する方法を説明します。

## How to create a W&B Experiment

W&B Experiment は次の 4 ステップで作成します:

1. [Initialize a W&B Run]({{< relref path="#initialize-a-wb-run" lang="ja" >}})
2. [Capture a dictionary of hyperparameters]({{< relref path="#capture-a-dictionary-of-hyperparameters" lang="ja" >}})
3. [Log metrics inside your training loop]({{< relref path="#log-metrics-inside-your-training-loop" lang="ja" >}})
4. [Log an artifact to W&B]({{< relref path="#log-an-artifact-to-wb" lang="ja" >}})

### Initialize a W&B run
[`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init" lang="ja" >}}) を使って W&B Run を作成します。

次の スニペット は、`"cat-classification"` という名前の W&B Project 内に run を作成し、この run を識別しやすくするために `"My first experiment"` という説明を付けます。さらに、将来の論文投稿用のベースライン 実験 であることを示すために、`"baseline"` と `"paper1"` のタグを付与します。

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
Note: `wandb.init()` を呼び出した時点で対象の Project が既に存在する場合、Run はその既存の Project に追加されます。たとえば、`"cat-classification"` という Project がすでにあるなら、その Project は削除されずそのまま使われ、新しい run がその Project に追加されます。
{{% /alert %}}

### Capture a dictionary of hyperparameters
学習率や モデル の種類などの ハイパーパラメーター の辞書を保存します。config に保存した モデル 設定は、後で 結果 を整理したりクエリしたりする際に役立ちます。

```python
with wandb.init(
    ...,
    config={"epochs": 100, "learning_rate": 0.001, "batch_size": 128},
) as run:
    ...
```

Experiment の設定方法の詳細は、[Configure Experiments]({{< relref path="./config.md" lang="ja" >}}) を参照してください。

### Log metrics inside your training loop
各トレーニング ステップの 精度 や損失などの メトリクス をログするには、[`run.log()`]({{< relref path="/ref/python/sdk/classes/run/#method-runlog" lang="ja" >}}) を呼び出します。

```python
model, dataloader = get_model(), get_data()

for epoch in range(run.config.epochs):
    for batch in dataloader:
        loss, accuracy = model.training_step()
        run.log({"accuracy": accuracy, "loss": loss})
```

W&B でログできるさまざまな データ 型の詳細は、[Log Data During Experiments]({{< relref path="/guides/models/track/log/" lang="ja" >}}) を参照してください。

### Log an artifact to W&B 
必要に応じて W&B Artifact をログします。Artifacts は Datasets や Models の バージョン 管理を簡単にします。
```python
# 任意のファイルやディレクトリーを保存できます。
# この例では、モデルに ONNX ファイルを出力する save() メソッドがあると仮定します。
model.save("path_to_model.onnx")
run.log_artifact("path_to_model.onnx", name="trained-model", type="model")
```
[Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) の詳細や、[Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) における Models の バージョン管理 について学びましょう。

### すべてを組み合わせる
上記のコードスニペットをまとめた完全な スクリプト は次のとおりです:
```python
import wandb

with wandb.init(
    project="cat-classification",
    notes="",
    tags=["baseline", "paper1"],
    # Run のハイパーパラメーターを記録します。
    config={"epochs": 100, "learning_rate": 0.001, "batch_size": 128},
) as run:
    # モデルとデータをセットアップします。
    model, dataloader = get_model(), get_data()

    # モデル性能を可視化するため、トレーニングしながらメトリクスをログします。
    for epoch in range(run.config["epochs"]):
        for batch in dataloader:
            loss, accuracy = model.training_step()
            run.log({"accuracy": accuracy, "loss": loss})

    # 学習済みモデルを Artifact としてアップロードします。
    model.save("path_to_model.onnx")
    run.log_artifact("path_to_model.onnx", name="trained-model", type="model")
```

## 次のステップ: 実験を可視化する 
W&B Dashboard を、機械学習 モデル の結果を整理・可視化するための中核として活用しましょう。数クリックで、[parallel coordinates plots]({{< relref path="/guides/models/app/features/panels/parallel-coordinates.md" lang="ja" >}}),[ パラメータの重要度 分析]({{< relref path="/guides/models/app/features/panels/parameter-importance.md" lang="ja" >}})、[その他のチャート]({{< relref path="/guides/models/app/features/panels/" lang="ja" >}}) など、豊富でインタラクティブな グラフ を作成できます。

{{< img src="/images/sweeps/quickstart_dashboard_example.png" alt="クイックスタート Sweeps ダッシュボードの例" >}}

experiments や特定の run の見方については、[Visualize results from experiments]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) を参照してください。

## ベストプラクティス
以下は、Experiments を作成する際に検討すべき推奨ガイドラインです:

1. **Run を終了する**: `with` 文内で `wandb.init()` を使い、コードが正常終了または例外で終了したタイミングで自動的に run を終了済みにします。
    * Jupyter ノートブック では、Run オブジェクトを自分で管理する方が便利な場合があります。この場合は、Run オブジェクトで `finish()` を明示的に呼び出して完了扱いにできます:

        ```python
        # ノートブックのセル内:
        run = wandb.init()

        # 別のセルで:
        run.finish()
        ```
2. **Config**: ハイパーパラメーター、アーキテクチャー、データセット、モデル の再現に使いたいその他の情報をトラックします。これらは列として表示されます。アプリで config の列を使って、run をグループ化、ソート、フィルターできます。
3. **Project**: Project は、比較可能な実験の集合です。各 Project には専用の ダッシュボード ページが用意され、異なる run グループをオン/オフして モデル の バージョン を比較できます。
4. **Notes**: スクリプトから直接、簡単なコミットメッセージを設定できます。W&B App の run の Overview セクションで、Notes を編集・参照できます。
5. **Tags**: ベースライン の run やお気に入りの run を識別します。タグで run をフィルタリングできます。タグは後から、W&B App の Project の ダッシュボード の Overview セクションで編集できます。
6. **実験を比較するために複数の run セットを作成する**: 実験を比較する際は、メトリクス を比較しやすくするために複数の run セットを作成しましょう。同じチャートやチャートのグループで run セットをオン/オフできます。

以下の コードスニペット は、上記のベストプラクティスを用いて W&B Experiment を定義する方法を示します:

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

W&B Experiment を定義する際に利用できるパラメータの詳細は、[API Reference Guide]({{< relref path="/ref/python/" lang="ja" >}}) の [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init" lang="ja" >}}) API ドキュメントを参照してください。