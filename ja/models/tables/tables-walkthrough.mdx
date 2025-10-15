---
title: 'チュートリアル: テーブルをログして、データを視覚化し、クエリする方法'
description: W&B Tables を 5 分で始めるクイックスタートで使い方を確認しましょう。
menu:
  default:
    identifier: ja-guides-models-tables-tables-walkthrough
    parent: tables
weight: 1
---

The following クイックスタート では、データ テーブルをログし、データを視覚化し、データをクエリする方法を示します。

以下のボタンを選択して、PyTorch クイックスタートの MNIST データの例のプロジェクト を試してください。

## 1. テーブルをログする
W&B を使用してテーブルをログします。新しいテーブルを作成するか、Pandas の Dataframe を渡すことができます。

{{< tabpane text=true >}}
{{% tab header="テーブルを作成する" value="construct" %}}
新しい Table を作成してログするには、次を使用します:
- [`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ja" >}}): [run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を作成して結果を追跡します。
- [`wandb.Table()`]({{< relref path="/ref/python/data-types/table.md" lang="ja" >}}): 新しいテーブル オブジェクトを作成します。
  - `columns`: カラム名を設定します。
  - `data`: 各行の内容を設定します。
- [`run.log()`]({{< relref path="/ref/python/log.md" lang="ja" >}}): テーブルをログし、W&B に保存します。

例はこちらです:
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
2. プロジェクト ワークスペースで run の名前を選択します。それぞれのユニークなテーブル キーに対して新しいパネルが追加されます。

{{< img src="/images/data_vis/wandb_demo_logged_sample_table.png" alt="" >}}

この例では、`my_table` がキー `"Table Name"` の下にログされています。

## 3. モデル バージョン を比較する

複数の W&B Runs のサンプル テーブルをログし、プロジェクト ワークスペースで結果を比較します。この [例のワークスペース](https://wandb.ai/carey/table-test?workspace=user-carey) では、複数の異なるバージョンから同じテーブルに行を結合する方法を示しています。

{{< img src="/images/data_vis/wandb_demo_toggle_on_and_off_cross_run_comparisons_in_tables.gif" alt="" >}}

モデルの結果を探索し評価するためにテーブルのフィルタ、ソート、グループ化の機能を使用してください。

{{< img src="/images/data_vis/wandb_demo_filter_on_a_table.png" alt="" >}}