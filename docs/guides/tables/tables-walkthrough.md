---
description: W&B Tables の使い方を 5 分で学ぶクイックスタートです。
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Walkthrough

次のクイックスタートでは、データテーブルのログ、データの可視化、データのクエリ方法を紹介します。

下のボタンを選択して、MNISTデータに関するPyTorchクイックスタートの例題プロジェクトを試してください。

## 1. テーブルのログ
W&Bでテーブルをログします。新しいテーブルを構築するか、Pandas DataFrameを渡すことができます。

<Tabs
  defaultValue="construct"
  values={[
    {label: 'Construct a table', value: 'construct'},
    {label: 'Pandas DataFrame', value: 'pandas'},
  ]}>
  <TabItem value="construct">

新しいTableを構築してログするには、以下を使用します:
- [`wandb.init()`](../../ref/python/init.md): 結果を追跡するための[run](../runs/intro.md)を作成します。
- [`wandb.Table()`](../../ref/python/data-types/table.md): 新しいテーブルオブジェクトを作成します。
  - `columns`: 列名を設定します。
  - `data`: 各行の内容を設定します。
- [`run.log()`](../../ref/python/log.md): テーブルをログしてW&Bに保存します。

例:
```python
import wandb

run = wandb.init(project="table-test")
# 新しいテーブルを作成してログします。
my_table = wandb.Table(columns=["a", "b"], data=[["a1", "b1"], ["a2", "b2"]])
run.log({"Table Name": my_table})
```
  </TabItem>
  <TabItem value="pandas">

Pandas DataFrameを`wandb.Table()`に渡して、新しいテーブルを作成します。

```python
import wandb
import pandas as pd

df = pd.read_csv("my_data.csv")

run = wandb.init(project="df-table")
my_table = wandb.Table(dataframe=df)
wandb.log({"Table Name": my_table})
```

サポートされているデータタイプの詳細については、W&B APIリファレンスガイドの[`wandb.Table`](../../ref/python/data-types/table.md)を参照してください。

  </TabItem>
</Tabs>

## 2. プロジェクトワークスペースでテーブルを可視化

ワークスペースで結果のテーブルを表示します。

1. W&Bアプリでプロジェクトに移動します。
2. プロジェクトワークスペースで実行の名前を選択します。各ユニークなテーブルキーに対して新しいパネルが追加されます。

![](/images/data_vis/wandb_demo_logged_sample_table.png)

この例では、`my_table`がキー`"Table Name"`の下にログされています。

## 3. モデルバージョン間での比較

複数のW&B Runsからサンプルテーブルをログし、プロジェクトワークスペースで結果を比較します。この[ワークスペースの例](https://wandb.ai/carey/table-test?workspace=user-carey)では、異なるバージョンからの行を同じテーブルに組み合わせる方法を示しています。

![](/images/data_vis/wandb_demo_toggle_on_and_off_cross_run_comparisons_in_tables.gif)

テーブルフィルター、ソート、グループ化機能を使用して、モデルの結果を探索および評価します。

![](/images/data_vis/wandb_demo_filter_on_a_table.png)