---
title: 実験を作成する
description: W&B 実験を作成します。
menu:
  default:
    identifier: ja-guides-models-track-launch
    parent: experiments
weight: 1
---

W&B Python SDKを使用して、機械学習実験をトラックします。その後、インタラクティブなダッシュボードで結果を確認するか、データをPythonにエクスポートしてプログラムでアクセスできます（[W&B Public API]({{< relref path="/ref/python/public-api/" lang="ja" >}})を参照）。

このガイドでは、W&Bのビルディングブロックを使用してW&B Experimentを作成する方法を説明します。

## W&B Experimentの作成方法

W&B Experimentを次の4つのステップで作成します：

1. [W&B Runを初期化]({{< relref path="#initialize-a-wb-run" lang="ja" >}})
2. [ハイパーパラメータの辞書をキャプチャ]({{< relref path="#capture-a-dictionary-of-hyperparameters" lang="ja" >}})
3. [トレーニングループ内でメトリクスをログ]({{< relref path="#log-metrics-inside-your-training-loop" lang="ja" >}})
4. [アーティファクトをW&Bにログ]({{< relref path="#log-an-artifact-to-wb" lang="ja" >}})

### W&B Runを初期化
[`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ja" >}})を使用してW&B Runを作成します。

以下のスニペットは、`“cat-classification”`という名前のW&Bプロジェクトで、`“My first experiment”`という説明を持つrunを作成し、これを識別するのに役立てます。タグ`“baseline”`と`“paper1”`は、このrunが将来的な論文の出版を意図したベースライン実験であることを思い出すために含まれています。

```python
import wandb

with wandb.init(
    project="cat-classification",
    notes="My first experiment",
    tags=["baseline", "paper1"],
) as run:
    ...
```

`wandb.init()`は、[Run]({{< relref path="/ref/python/run.md" lang="ja" >}})オブジェクトを返します。

{{% alert %}}
注意: `wandb.init()`を呼び出したときにプロジェクトが既に存在している場合、Runは既存のプロジェクトに追加されます。例えば、既に`“cat-classification”`という名前のプロジェクトがある場合、そのプロジェクトは既存のままで削除されず、新しいrunがそのプロジェクトに追加されます。
{{% /alert %}}

### ハイパーパラメータの辞書をキャプチャ
学習率やモデルタイプといったハイパーパラメータの辞書を保存します。設定でキャプチャしたモデル設定は、後で結果の整理やクエリに役立ちます。

```python
with wandb.init(
    ...,
    config={"epochs": 100, "learning_rate": 0.001, "batch_size": 128},
) as run:
    ...
```

実験を設定する方法の詳細については、[Configure Experiments]({{< relref path="./config.md" lang="ja" >}})を参照してください。

### トレーニングループ内でメトリクスをログ
[`run.log()`]({{< relref path="/ref/python/log.md" lang="ja" >}})を呼び出して、精度や損失といった各トレーニングステップに関するメトリクスをログします。

```python
model, dataloader = get_model(), get_data()

for epoch in range(run.config.epochs):
    for batch in dataloader:
        loss, accuracy = model.training_step()
        run.log({"accuracy": accuracy, "loss": loss})
```

W&Bでログできるさまざまなデータタイプの詳細については、[Log Data During Experiments]({{< relref path="/guides/models/track/log/" lang="ja" >}})を参照してください。

### アーティファクトをW&Bにログ
オプションでW&Bアーティファクトをログします。アーティファクトは、データセットやモデルのバージョン管理を容易にします。
```python
# 任意のファイルまたはディレクトリを保存できます。この例では、モデルがONNXファイルを出力するsave()メソッドを持つと仮定しています。
model.save("path_to_model.onnx")
run.log_artifact("path_to_model.onnx", name="trained-model", type="model")
```
[Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}})や[Registry]({{< relref path="/guides/core/registry/" lang="ja" >}})でのモデルのバージョン管理について詳しく学んでください。

### すべてをまとめる
前述のコードスニペットを使った完全なスクリプトは以下の通りです：
```python
import wandb

with wandb.init(
    project="cat-classification",
    notes="",
    tags=["baseline", "paper1"],
    # runのハイパーパラメータを記録
    config={"epochs": 100, "learning_rate": 0.001, "batch_size": 128},
) as run:
    # モデルとデータをセットアップ
    model, dataloader = get_model(), get_data()

    # モデルのパフォーマンスを可視化するためにメトリクスをログしながらトレーニングを実行
    for epoch in range(run.config["epochs"]):
        for batch in dataloader:
            loss, accuracy = model.training_step()
            run.log({"accuracy": accuracy, "loss": loss})

    # 訓練済みモデルをアーティファクトとしてアップロード
    model.save("path_to_model.onnx")
    run.log_artifact("path_to_model.onnx", name="trained-model", type="model")
```

## 次のステップ：実験を可視化
W&Bダッシュボードを使用して、機械学習モデルの結果を整理し可視化する中央の場所として利用します。[parallel coordinates plots]({{< relref path="/guides/models/app/features/panels/parallel-coordinates.md" lang="ja" >}})、[parameter importance analyzes]({{< relref path="/guides/models/app/features/panels/parameter-importance.md" lang="ja" >}})、および[その他の]({{< relref path="/guides/models/app/features/panels/" lang="ja" >}})ようなリッチでインタラクティブなグラフを数クリックで構築します。

{{< img src="/images/sweeps/quickstart_dashboard_example.png" alt="クイックスタートスイープダッシュボードの例" >}}

実験や特定runの表示方法についての詳細は、[Visualize results from experiments]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}})を参照してください。

## ベストプラクティス
以下は、実験を作成する際に考慮すべきいくつかのガイドラインです：

1. **Runを終了させる**: `wandb.init()`を`with`文で使用して、コードが完了したときや例外が発生したときにrunを自動的に終了させます。
    * Jupyterノートブックでは、Runオブジェクトを自分で管理する方が便利な場合があります。この場合、Runオブジェクトで`finish()`を明示的に呼び出して完了をマークできます：

        ```python
        # ノートブックセル内で：
        run = wandb.init()

        # 別のセルで：
        run.finish()
        ```
2. **Config**: ハイパーパラメータ、アーキテクチャー、データセット、およびモデルの再現に使用したい他のすべてのものをトラックします。これらは列に表示され、アプリ内で動的にrunをグループ化、並べ替え、およびフィルタリングするためにconfig列を使用します。
3. **プロジェクト**: プロジェクトは比較可能な一連の実験です。各プロジェクトは専用のダッシュボードページを持ち、異なるモデルバージョンを比較するrunの異なるグループを簡単にオンオフできます。
4. **ノート**: スクリプトから直接クイックコミットメッセージを設定します。ノートを編集し、W&Bアプリのrunのオーバービューセクションでアクセスします。
5. **タグ**: ベースラインrunとお気に入りのrunを識別します。タグを使用してrunをフィルタリングできます。タグは後で、プロジェクトのダッシュボードのオーバービューセクションで編集できます。
6. **実験を比較するために複数のrunセットを作成**: 実験を比較するときは、メトリクスを比較しやすくするために複数のrunセットを作成します。runセットを同じグラフまたは一連のグラフでオンオフできます。

以下のコードスニペットは、上記のベストプラクティスを使用してW&B Experimentを定義する方法を示しています：

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

W&B Experimentを定義する際に利用可能なパラメータの詳細については、[`wandb.init`]({{< relref path="/ref/python/init.md" lang="ja" >}}) APIドキュメントを[APIリファレンスガイド]({{< relref path="/ref/python/" lang="ja" >}})で参照してください。