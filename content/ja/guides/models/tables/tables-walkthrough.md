---
title: チュートリアル：テーブルのログ、データの可視化とクエリ
description: 5分でできるクイックスタートで、W&B テーブル の使い方を学びましょう。
menu:
  default:
    identifier: ja-guides-models-tables-tables-walkthrough
    parent: tables
weight: 1
---

以下のクイックスタートでは、データ テーブルのログ方法、データの可視化、データのクエリ方法を解説します。

下のボタンを選択して、MNIST データ上で PyTorch クイックスタートのサンプルプロジェクトを試してみましょう。

## 1. テーブルをログする

W&B でテーブルをログします。新しくテーブルを作成するか、Pandas DataFrame を渡すことができます。

{{< tabpane text=true >}}
{{% tab header="テーブルを作成する" value="construct" %}}
新しい Table を作成してログするには、以下を使用します:
- [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}): [run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を作成し、結果を追跡します。
- [`wandb.Table()`]({{< relref path="/ref/python/sdk/data-types/table.md" lang="ja" >}}): 新しいテーブルオブジェクトを作成します。
  - `columns`: カラム名を設定します。
  - `data`: 各行の内容を設定します。
- [`wandb.Run.log()`]({{< relref path="/ref/python/sdk/classes/run.md/#method-runlog" lang="ja" >}}): テーブルを W&B に保存します。

例を紹介します:
```python
import wandb

with wandb.init(project="table-test") as run:
    # 新しいテーブルを作成してログする
    my_table = wandb.Table(columns=["a", "b"], data=[["a1", "b1"], ["a2", "b2"]])
    run.log({"Table Name": my_table})
```
{{% /tab %}}

{{% tab header="Pandas DataFrame" value="pandas"%}}
Pandas DataFrame を `wandb.Table()` に渡して新しいテーブルを作成します。

```python
import wandb
import pandas as pd

df = pd.read_csv("my_data.csv")

with wandb.init(project="df-table") as run:
    # DataFrame から新しいテーブルを作成し、W&B にログする
  my_table = wandb.Table(dataframe=df)
  run.log({"Table Name": my_table})
```

対応しているデータ型の詳細については、W&B API リファレンスガイドの [`wandb.Table`]({{< relref path="/ref/python/sdk/data-types/table.md" lang="ja" >}}) をご覧ください。
{{% /tab %}}
{{< /tabpane >}}

## 2. プロジェクト Workspace でテーブルを可視化する

作成したテーブルを自分の workspace で表示できます。

1. W&B App で対象プロジェクトへ移動します。
2. プロジェクト workspace で run の名前を選択します。ユニークなテーブル キーごとに新しいパネルが追加されます。

{{< img src="/images/data_vis/wandb_demo_logged_sample_table.png" alt="サンプルテーブルをログした画面" >}}

この例では `my_table` が `"Table Name"` というキーでログされています。

## 3. モデル バージョンを比較する

複数の W&B Run からサンプル テーブルをログし、プロジェクト workspace で結果を比較しましょう。この [サンプル workspace](https://wandb.ai/carey/table-test?workspace=user-carey) では、複数バージョンから行をひとつのテーブルにまとめて比較しています。

{{< img src="/images/data_vis/wandb_demo_toggle_on_and_off_cross_run_comparisons_in_tables.gif" alt="Run をまたいだテーブル比較" >}}

テーブルのフィルター、ソート、グループ化などの機能を使い、モデルの結果を探索・評価できます。

{{< img src="/images/data_vis/wandb_demo_filter_on_a_table.png" alt="テーブルのフィルタリング" >}}