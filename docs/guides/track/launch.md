---
description: W&Bの実験を作成する。
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# 実験の作成

<head>
  <title>W&B 実験の開始</title>
</head>

W&B Python SDK を使用して機械学習実験を追跡します。その後、インタラクティブなダッシュボードで結果を確認したり、[W&B Public API](../../ref/python/public-api/README.md)でプログラム的にアクセスするためにデータをエクスポートすることができます。

このガイドでは、W&B のビルディングブロックを使用して W&B Experiment を作成する方法について説明します。

## W&B Experimentの作成方法

W&B Experimentを以下の4ステップで作成します：

1. [W&B Run の初期化](#initialize-a-wb-run)
2. [ハイパーパラメータの辞書をキャプチャ](#capture-a-dictionary-of-hyperparameters)
3. [トレーニングループ内でメトリクスをログ](#log-metrics-inside-your-training-loop)
4. [W&B にアーティファクトをログ](#log-an-artifact-to-wb)

### W&B Run の初期化
スクリプトの最初で、[`wandb.init()`](../../ref/python/init.md) API を呼び出し、バックグラウンドプロセスを生成して W&B Run としてデータを同期・ログします。

以下のコードスニペットは、新しい W&B プロジェクト `“cat-classification”` を作成する方法を示しています。この run を識別するために `“My first experiment”` というノートが追加されました。タグ `“baseline”` と `“paper1”` が含まれており、この run が将来の論文発表を目指したベースライン実験であることを示しています。

```python
# W&B Pythonライブラリをインポート
import wandb

# 1. W&B Runの開始
run = wandb.init(
    project="cat-classification",
    notes="My first experiment",
    tags=["baseline", "paper1"],
)
```
W&B を `wandb.init()` で初期化すると、[Run](../../ref/python/run.md) オブジェクトが返されます。さらに、W&B はすべてのログとファイルを保存し、非同期で W&B サーバーにストリームするローカルディレクトリを作成します。

:::info
注: run は、wandb.init() を呼び出した時点で既存のプロジェクトに追加されます。例えば、すでに `“cat-classification”` というプロジェクトがある場合、そのプロジェクトは削除されずに続行され、新しい run がそのプロジェクトに追加されます。
:::

### ハイパーパラメータの辞書をキャプチャ
学習率やモデルタイプなどのハイパーパラメータの辞書を保存します。config にキャプチャされたモデル設定は、後で結果を整理したりクエリを実行する際に便利です。

```python
# 2. ハイパーパラメータの辞書をキャプチャ
wandb.config = {"epochs": 100, "learning_rate": 0.001, "batch_size": 128}
```
実験の設定方法については、[Configure Experiments](./config.md) を参照してください。

### トレーニングループ内でメトリクスをログ
各 `for` ループ（エポック）中にメトリクスをログします。精度と損失値が計算され、[`wandb.log()`](../../ref/python/log.md) を使用して W&B にログされます。デフォルトでは、wandb.log を呼び出すと新しいステップが履歴オブジェクトに追加され、サマリーオブジェクトが更新されます。

以下のコード例では、`wandb.log` を使用してメトリクスをログする方法を示しています。

:::note
モードの設定とデータの取得方法の詳細は省略しています。
:::

```python
# モデルとデータの設定
model, dataloader = get_model(), get_data()

for epoch in range(wandb.config.epochs):
    for batch in dataloader:
        loss, accuracy = model.training_step()
        # 3. トレーニングループ内でメトリクスをログして
        # モデル性能を可視化
        wandb.log({"accuracy": accuracy, "loss": loss})
```
W&B でログできるさまざまなデータタイプの詳細については、[Log Data During Experiments](./log/intro.md) を参照してください。

### W&B にアーティファクトをログ
オプションで W&B Artifact をログします。Artifacts により、データセットとモデルのバージョン管理が簡単になります。

```python
wandb.log_artifact(model)
```
Artifacts についての詳細は [Artifacts Chapter](../artifacts/intro.md) を参照してください。モデルのバージョン管理については [Model Management](../model_registry/intro.md) を参照してください。

### まとめ
以下は、前述のコードスニペットを全て含んだフルスクリプトです：

```python
# W&B Pythonライブラリをインポート
import wandb

# 1. W&B Runの開始
run = wandb.init(project="cat-classification", notes="", tags=["baseline", "paper1"])

# 2. ハイパーパラメータの辞書をキャプチャ
wandb.config = {"epochs": 100, "learning_rate": 0.001, "batch_size": 128}

# モデルとデータの設定
model, dataloader = get_model(), get_data()

for epoch in range(wandb.config.epochs):
    for batch in dataloader:
        loss, accuracy = model.training_step()
        # 3. トレーニングループ内でメトリクスをログして
        # モデル性能を可視化
        wandb.log({"accuracy": accuracy, "loss": loss})

# 4. W&B にアーティファクトをログ
wandb.log_artifact(model)

# オプション：最後にモデルを保存
model.to_onnx()
wandb.save("model.onnx")
```

## 次のステップ: 実験を可視化する
W&B ダッシュボードを使用して、機械学習モデルの結果を中央で組織し、可視化します。数回のクリックでリッチなインタラクティブチャートを作成できます。例えば、[parallel coordinates plots](../app/features/panels/parallel-coordinates.md)、[parameter importance analyzes](../app/features/panels/parameter-importance.md)、および[その他](../app/features/panels/intro.md)などです。

![Quickstart Sweeps Dashboard example](/images/sweeps/quickstart_dashboard_example.png)

実験と特定の run の表示方法について詳しくは、[Visualize results from experiments](./app.md) を参照してください。

## ベストプラクティス
以下は、実験を作成する際の推奨ガイドラインです：

1. **Config**: ハイパーパラメータ、アーキテクチャー、データセット、およびモデルを再現するために使用したいものを追跡します。これらは列として表示され、アプリ内でコンフィグ列を使用して run を動的にグループ化、ソート、フィルターできます。
2. **Project**: 比較可能な実験のセットです。各プロジェクトには専用のダッシュボードページがあり、異なるモデルバージョンを比較するために異なるグループの run を簡単にオン/オフできます。
3. **Notes**: 自分への簡単なコミットメッセージです。スクリプトからノートを設定できます。ノートは W&B アプリのプロジェクトダッシュボードの概要セクションで後から編集できます。
4. **Tags**: ベースライン run やお気に入りの run を識別します。タグを使用して run をフィルタリングできます。タグは W&B アプリのプロジェクトダッシュボードの概要セクションで後から編集できます。

以下のコードスニペットは、上記のベストプラクティスに従って W&B Experiment を定義する方法を示しています：

```python
import wandb

config = dict(
    learning_rate=0.01, momentum=0.2, architecture="CNN", dataset_id="cats-0192"
)

wandb.init(
    project="detect-cats",
    notes="tweak baseline",
    tags=["baseline", "paper1"],
    config=config,
)
```

W&B Experimentを定義する際の使用可能なパラメータの詳細については、[`wandb.init`](../../ref/python/init.md) API ドキュメントを [API Reference Guide](../../ref/python/README.md) で参照してください。