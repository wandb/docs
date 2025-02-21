---
title: 'Tutorial: Log tables, visualize and query data'
description: W&B テーブル を使用する方法を、この 5 分間の クイックスタート で見てみましょう。
menu:
  default:
    identifier: ja-guides-core-tables-tables-walkthrough
    parent: tables
weight: 1
---

次の クイックスタート では、データテーブル のログ記録、データの可視化、およびデータのクエリ方法について説明します。

下のボタンを選択して、MNIST データに関する PyTorch の クイックスタート のサンプル プロジェクト を試してください。

## 1. テーブル のログ
W&B で テーブル を ログ します。新しい テーブル を作成するか、Pandas Dataframe を渡すことができます。

{{< tabpane text=true >}}
{{% tab header="テーブル の作成" value="construct" %}}
新しい テーブル を作成して ログ に記録するには、以下を使用します。
- [`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ja" >}}): [run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を作成して、結果を追跡します。
- [`wandb.Table()`]({{< relref path="/ref/python/data-types/table.md" lang="ja" >}}): 新しい テーブル オブジェクトを作成します。
  - `columns`: 列名を設定します。
  - `data`: 各行の内容を設定します。
- [`run.log()`]({{< relref path="/ref/python/log.md" lang="ja" >}}): テーブル を ログ に記録して、W&B に保存します。

次に例を示します。
```python
import wandb

run = wandb.init(project="table-test")
# 新しいテーブルを作成してログに記録します。
my_table = wandb.Table(columns=["a", "b"], data=[["a1", "b1"], ["a2", "b2"]])
run.log({"Table Name": my_table})
```
{{% /tab %}}

{{% tab header="Pandas Dataframe" value="pandas"%}}
Pandas Dataframe を `wandb.Table()` に渡して、新しい テーブル を作成します。

```python
import wandb
import pandas as pd

df = pd.read_csv("my_data.csv")

run = wandb.init(project="df-table")
my_table = wandb.Table(dataframe=df)
wandb.log({"Table Name": my_table})
```

サポートされているデータ型 の詳細については、W&B API Reference Guide の [`wandb.Table`]({{< relref path="/ref/python/data-types/table.md" lang="ja" >}}) を参照してください。
{{% /tab %}}
{{< /tabpane >}}

## 2. プロジェクト の ワークスペース で テーブル を可視化する

ワークスペース で結果の テーブル を表示します。

1. W&B App で プロジェクト に移動します。
2. プロジェクト の ワークスペース で run の名前を選択します。一意の テーブル の キー ごとに新しい パネル が追加されます。

{{< img src="/images/data_vis/wandb_demo_logged_sample_table.png" alt="" >}}

この例では、`my_table` は キー `"Table Name"` で ログ に記録されます。

## 3. モデル の バージョン 間で比較する

複数の W&B の Runs からサンプル テーブル を ログ に記録し、 プロジェクト の ワークスペース で結果を比較します。この[ワークスペース の例](https://wandb.ai/carey/table-test?workspace=user-carey)では、同じ テーブル 内の複数の異なる バージョン の行を結合する方法を示します。

{{< img src="/images/data_vis/wandb_demo_toggle_on_and_off_cross_run_comparisons_in_tables.gif" alt="" >}}

テーブル のフィルター、ソート、およびグループ化機能を使用して、モデル の結果を調べて評価します。

{{< img src="/images/data_vis/wandb_demo_filter_on_a_table.png" alt="" >}}
