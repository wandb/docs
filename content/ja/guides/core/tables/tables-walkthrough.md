---
title: 'Tutorial: Log tables, visualize and query data'
description: W&B Tables を使い始めるには、この5分間のクイックスタートを参照してください。
menu:
  default:
    identifier: ja-guides-core-tables-tables-walkthrough
    parent: tables
weight: 1
---

次のクイックスタートでは、データテーブルのログ、データの視覚化、およびデータのクエリを実演します。

以下のボタンを選択して、MNIST データを使用した PyTorch クイックスタートの例のプロジェクトを試してください。

## 1. テーブルをログする
W&B を使用してテーブルをログします。新しいテーブルを構築するか、Pandas Dataframe を渡すことができます。

{{< tabpane text=true >}}
{{% tab header="テーブルを構築する" value="construct" %}}
新しいテーブルを構築してログするには、以下を使用します。
- [`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ja" >}}): 結果を追跡する[run]({{< relref path="/guides/models/track/runs/" lang="ja" >}})を作成します。
- [`wandb.Table()`]({{< relref path="/ref/python/data-types/table.md" lang="ja" >}}): 新しいテーブルのオブジェクトを作成します。
  - `columns`: 列名を設定します。
  - `data`: 各行のコンテンツを設定します。
- [`run.log()`]({{< relref path="/ref/python/log.md" lang="ja" >}}): テーブルをログし、W&B に保存します。

例は以下の通りです。
```python
import wandb

run = wandb.init(project="table-test")
# 新しいテーブルを作成してログします。
my_table = wandb.Table(columns=["a", "b"], data=[["a1", "b1"], ["a2", "b2"]])
run.log({"Table Name": my_table})
```
{{% /tab %}}

{{% tab header="Pandas Dataframe" value="pandas"%}}
Pandas Dataframe を `wandb.Table()` に渡して、新しいテーブルを作成します。

```python
import wandb
import pandas as pd

df = pd.read_csv("my_data.csv")

run = wandb.init(project="df-table")
my_table = wandb.Table(dataframe=df)
wandb.log({"Table Name": my_table})
```

サポートされているデータ型の詳細については、W&B API リファレンス ガイドの [`wandb.Table`]({{< relref path="/ref/python/data-types/table.md" lang="ja" >}}) を参照してください。
{{% /tab %}}
{{< /tabpane >}}

## 2. プロジェクトワークスペースでテーブルを視覚化する

ワークスペースで結果のテーブルを表示します。

1. W&B アプリでプロジェクトに移動します。
2. プロジェクトワークスペースで run の名前を選択します。ユニークなテーブルキーごとに新しいパネルが追加されます。

{{< img src="/images/data_vis/wandb_demo_logged_sample_table.png" alt="" >}}

この例では、`my_table` がキー `"Table Name"` の下にログされています。

## 3. モデルバージョン間で比較する

複数の W&B Runs からサンプルテーブルをログし、プロジェクトワークスペースで結果を比較します。この[例のワークスペース](https://wandb.ai/carey/table-test?workspace=user-carey)では、異なるバージョンからの行を同じテーブルに組み合わせる方法を示しています。

{{< img src="/images/data_vis/wandb_demo_toggle_on_and_off_cross_run_comparisons_in_tables.gif" alt="" >}}

テーブルフィルタ、ソート、グループ化機能を使用して、モデルの結果を探索および評価します。

{{< img src="/images/data_vis/wandb_demo_filter_on_a_table.png" alt="" >}}