---
description: Traverse automatically created direct acyclic W&B Artifact graphs.
displayed_sidebar: ja
---

# アーティファクトグラフの探索とたどり

<head>
    <title>有向非巡回W＆Bアーティファクトグラフを探索しましょう。</title>
</head>
Weights & Biasesは、特定のrunでログしたアーティファクトと、特定のrunが使用したアーティファクトを自動的にトラッキングします。W&B App UIまたはプログラムでアーティファクトの履歴を調べてみましょう。

## W&B App UIでアーティファクトをたどる

グラフビューでは、開発フローの概要が表示されます。
アーティファクトのグラフを表示する方法：

1. W&BアプリのUIでプロジェクトに移動します。
2. 左パネルのアーティファクトアイコンを選択します。
3. **履歴**を選択します。

runやアーティファクトを作成する際に提供する`type`が、グラフの作成に使用されます。runやアーティファクトの入力と出力は、矢印でグラフに表示されます。アーティファクトは青い四角形で表され、Runは緑の四角形で表されます。
提供されるアーティファクトのタイプは、**ARTIFACT**ラベルの隣にある濃い青のヘッダーにあります。アーティファクトの名前とアーティファクトのバージョンは、**ARTIFACT**ラベルの下の淡い青の領域に表示されます。

runを初期化する際に提供するジョブタイプは、**RUN**ラベルの隣にあります。W&Bのrun名は、**RUN**ラベルの下の淡い緑の領域にあります。

:::info
アーティファクトのタイプと名前は、左のサイドバーと**Lineage**タブの両方で表示することができます。
:::
例えば、上の画像では、アーティファクトは "raw_dataset" というタイプで定義されています（ピンクの四角）。アーティファクトの名前は "MNIST_raw"（ピンクの線）と呼ばれています。そのアーティファクトは、トレーニングに使用されました。トレーニングのrunの名前は "vivid-snow-42" と呼ばれています。そのrunは、"mnist-19pofeku" という名前の "モデル"アーティファクト（オレンジの四角）を生成しました。

![実験に使用されたアーティファクトとrunsのDAGビュー](/images/artifacts/example_dag_with_sidebar.png)
詳細なビューには、ダッシュボードの左上にある **Explode** トグルを選択してください。拡張されたグラフでは、プロジェクト内のすべてのrunやアーティファクトの詳細を確認できます。この[example Graph page](https://wandb.ai/shawn/detectron2-11/artifacts/dataset/furniture-small-val/v0/lineage)で自分で試してみてください。

## アーティファクトをプログラム的にたどる

W&B Public API（[wandb.Api](https://docs.wandb.ai/ref/python/public-api/api)）を使用して、アーティファクトオブジェクトを作成します。プロジェクト名、アーティファクト名、およびアーティファクトのエイリアスを指定してください。

```python
import wandb

api = wandb.Api()

artifact = api.artifact('プロジェクト/アーティファクト:エイリアス')
```
アーティファクトからグラフを辿るために、アーティファクトオブジェクトの[`logged_by`](https://docs.wandb.ai/ref/python/public-api/artifact#logged\_by)メソッドと[`used_by`](https://docs.wandb.ai/ref/python/public-api/artifact#used\_by)メソッドを使用します。

```python
# アーティファクトからグラフの上下を辿る：
producer_run = artifact.logged_by()
consumer_runs = artifact.used_by()
```
#### runからトラバースする

W&B Public API（[wandb.Api.Run](https://docs.wandb.ai/ref/python/public-api/run)）を使用して、アーティファクトオブジェクトを作成します。エンティティ名、プロジェクト名、およびRun IDを指定してください。

```python
import wandb
api = wandb.Api()

artifact = api.run('エンティティ/プロジェクト/run_id')
```

与えられたrunからグラフをたどるために、[`logged_artifacts`](https://docs.wandb.ai/ref/python/public-api/run#logged_artifacts) および [`used_artifacts`](https://docs.wandb.ai/ref/python/public-api/run#used_artifacts) メソッドを使用します。

```python
# runからグラフを上下に辿る：
logged_artifacts = run.logged_artifacts()
used_artifacts = run.used_artifacts()
```