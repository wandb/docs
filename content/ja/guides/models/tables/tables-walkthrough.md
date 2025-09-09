---
title: 'チュートリアル: テーブルを ログし、データを 可視化・クエリする'
description: この 5 分のクイックスタートで W&B テーブルの使い方を学びましょう。
menu:
  default:
    identifier: ja-guides-models-tables-tables-walkthrough
    parent: tables
weight: 1
---

以下の クイックスタート では、データ テーブル の ログ、データの可視化、データのクエリ方法を紹介します。

下のボタンを選択して、MNIST データ を用いた PyTorch クイックスタート のサンプル プロジェクト を試してください。 

## 1. テーブル を ログ する
W&B で テーブル を ログ します。新しい テーブル を作成するか、Pandas の DataFrame を渡します。

{{< tabpane text=true >}}
{{% tab header="テーブルを作成" value="construct" %}}
新しい テーブル を作成して ログ するには、次を使用します:
- [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}): 結果 を追跡するための [run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を作成します。
- [`wandb.Table()`]({{< relref path="/ref/python/sdk/data-types/table.md" lang="ja" >}}): 新しい テーブル オブジェクト を作成します。
  - `columns`: 列名を設定します。
  - `data`: 各行の内容を設定します。
- [`wandb.Run.log()`]({{< relref path="/ref/python/sdk/classes/run.md/#method-runlog" lang="ja" >}}): テーブル を ログ して W&B に保存します。

使用例:
```python
import wandb

with wandb.init(project="table-test") as run:
    # 新しいテーブルを作成してログします。
    my_table = wandb.Table(columns=["a", "b"], data=[["a1", "b1"], ["a2", "b2"]])
    run.log({"Table Name": my_table})
```
{{% /tab %}}

{{% tab header="Pandas DataFrame" value="pandas"%}}
Pandas の DataFrame を `wandb.Table()` に渡して、新しい テーブル を作成します。

```python
import wandb
import pandas as pd

df = pd.read_csv("my_data.csv")

with wandb.init(project="df-table") as run:
    # DataFrame から新しいテーブルを作成して
    # W&B にログします。
  my_table = wandb.Table(dataframe=df)
  run.log({"Table Name": my_table})
```

サポートされているデータ型の詳細は、W&B API リファレンス ガイドの [`wandb.Table`]({{< relref path="/ref/python/sdk/data-types/table.md" lang="ja" >}}) を参照してください。
{{% /tab %}}
{{< /tabpane >}}


## 2. プロジェクト の ワークスペース で テーブル を可視化する

ワークスペース で 結果 の テーブル を表示します。 

1. W&B App で自分の プロジェクト に移動します。
2. プロジェクト の ワークスペース で run 名を選択します。一意の テーブル キー ごとに新しい パネル が追加されます。 

{{< img src="/images/data_vis/wandb_demo_logged_sample_table.png" alt="サンプルの テーブル の ログ" >}}

この例では、`my_table` は キー "Table Name" の下に ログ されています。

## 3. モデル の バージョン 間で比較する

複数の W&B Runs からテーブルのサンプルを ログ し、プロジェクト の ワークスペース で 結果 を比較します。この [example workspace](https://wandb.ai/carey/table-test?workspace=user-carey) では、同じ テーブル 内で複数の異なる バージョン の行を結合する方法を示します。

{{< img src="/images/data_vis/wandb_demo_toggle_on_and_off_cross_run_comparisons_in_tables.gif" alt="run をまたいだ テーブル 比較" >}}

テーブル のフィルタ、並べ替え、グループ化機能を使って、モデルの 結果 を探索・評価します。

{{< img src="/images/data_vis/wandb_demo_filter_on_a_table.png" alt="テーブル のフィルタリング" >}}