---
title: チュートリアル：テーブルをログし、データを可視化・クエリする
description: 5分でできるクイックスタートで W&B テーブルの使い方を学びましょう。
menu:
  default:
    identifier: tables-walkthrough
    parent: tables
weight: 1
---

以下のクイックスタートでは、データ テーブルのログ、データの可視化、データのクエリ方法を説明します。

下のボタンから、MNIST データを使った PyTorch クイックスタートのサンプルプロジェクトをお試しください。

## 1. テーブルのログ
W&B にテーブルをログします。新しいテーブルを作成することも、Pandas の DataFrame を渡すこともできます。

{{< tabpane text=true >}}
{{% tab header="テーブルを作成" value="construct" %}}
新しい Table を作成してログするには、次のものを使用します:
- [`wandb.init()`]({{< relref "/ref/python/sdk/functions/init.md" >}}): 結果を追跡する [run]({{< relref "/guides/models/track/runs/" >}}) を作成します。
- [`wandb.Table()`]({{< relref "/ref/python/sdk/data-types/table.md" >}}): 新しいテーブルオブジェクトを作成します。
  - `columns`: 列名を設定します。
  - `data`: 各行の内容を設定します。
- [`wandb.Run.log()`]({{< relref "/ref/python/sdk/classes/run.md/#method-runlog" >}}): テーブルをログして W&B に保存します。

例:

```python
import wandb

with wandb.init(project="table-test") as run:
    # 新しいテーブルを作成してログします。
    my_table = wandb.Table(columns=["a", "b"], data=[["a1", "b1"], ["a2", "b2"]])
    run.log({"Table Name": my_table})
```
{{% /tab %}}

{{% tab header="Pandas DataFrame" value="pandas"%}}
Pandas の DataFrame を `wandb.Table()` に渡して新しいテーブルを作成します。

```python
import wandb
import pandas as pd

df = pd.read_csv("my_data.csv")

with wandb.init(project="df-table") as run:
    # DataFrame から新しいテーブルを作成
    # して、それを W&B にログします。
  my_table = wandb.Table(dataframe=df)
  run.log({"Table Name": my_table})
```

サポートされているデータ型の詳細は、W&B API リファレンスガイドの [`wandb.Table`]({{< relref "/ref/python/sdk/data-types/table.md" >}}) をご覧ください。
{{% /tab %}}
{{< /tabpane >}}


## 2. プロジェクト ワークスペースでテーブルを可視化

run の結果として作成されたテーブルを workspace で表示します。

1. W&B アプリで自分のプロジェクトに移動します。
2. プロジェクトワークスペースで run の名前を選択します。各テーブルキーごとに新しいパネルが追加されます。

{{< img src="/images/data_vis/wandb_demo_logged_sample_table.png" alt="Sample table logged" >}}

この例では、`my_table` は `"Table Name"` というキーでログされています。

## 3. モデルの複数バージョンを比較

複数の W&B Run からサンプルテーブルをログし、プロジェクトワークスペースでその結果を比較できます。この [サンプル ワークスペース](https://wandb.ai/carey/table-test?workspace=user-carey) では、異なるバージョンの行を同じテーブルにまとめて表示しています。

{{< img src="/images/data_vis/wandb_demo_toggle_on_and_off_cross_run_comparisons_in_tables.gif" alt="Cross-run table comparison" >}}

テーブルのフィルターやソート、グループ化機能を使って、モデルの結果を探索・評価しましょう。

{{< img src="/images/data_vis/wandb_demo_filter_on_a_table.png" alt="Table filtering" >}}