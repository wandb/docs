---
description: "W&B Tables\u306E\u4F7F\u3044\u65B9\u30925\u5206\u3067\u5B66\u3079\u308B\
  \u30AF\u30A4\u30C3\u30AF\u30B9\u30BF\u30FC\u30C8\u3002"
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# ウォークスルー

以下のクイックスタートでは、データテーブルのログ、データの可視化、およびデータのクエリ方法を示します。

以下のボタンを選択して、MNISTデータに関するPyTorchクイックスタートの例プロジェクトを試してみてください。

## 1. テーブルをログする
W&Bでテーブルをログします。新しいテーブルを構築するか、Pandas DataFrameを渡すことができます。

<Tabs
  defaultValue="construct"
  values={[
    {label: 'テーブルを構築する', value: 'construct'},
    {label: 'Pandas DataFrame', value: 'pandas'},
  ]}>
  <TabItem value="construct">

新しいTableを構築してログするには、以下を使用します:
- [`wandb.init()`](../../ref/python/init.md): 結果を追跡するための[run](../runs/intro.md)を作成します。
- [`wandb.Table()`](../../ref/python/data-types/table.md): 新しいテーブルオブジェクトを作成します。
  - `columns`: 列名を設定します。
  - `data`: 各行の内容を設定します。
- [`run.log()`](../../ref/python/log.md): テーブルをログしてW&Bに保存します。

以下は例です:
```python
import wandb

run = wandb.init(project="table-test")
# 新しいテーブルを作成してログする。
my_table = wandb.Table(columns=["a", "b"], data=[["a1", "b1"], ["a2", "b2"]])
run.log({"Table Name": my_table})
```
  </TabItem>
  <TabItem value="pandas">

Pandas DataFrameを`wandb.Table()`に渡して新しいテーブルを作成します。

```python
import wandb
import pandas as pd

df = pd.read_csv("my_data.csv")

run = wandb.init(project="df-table")
my_table = wandb.Table(dataframe=df)
wandb.log({"Table Name": my_table})
```

サポートされているデータ型の詳細については、W&B APIリファレンスガイドの[`wandb.Table`](../../ref/python/data-types/table.md)を参照してください。

  </TabItem>
</Tabs>

## 2. プロジェクトワークスペースでテーブルを可視化する

ワークスペースで結果のテーブルを表示します。

1. W&Bアプリでプロジェクトに移動します。
2. プロジェクトワークスペースでrunの名前を選択します。各ユニークなテーブルキーに対して新しいパネルが追加されます。

![](/images/data_vis/wandb_demo_logged_sample_table.png)

この例では、`my_table`がキー `"Table Name"`の下にログされています。

## 3. モデルバージョン間で比較する

複数のW&B Runsからサンプルテーブルをログし、プロジェクトワークスペースで結果を比較します。この[例のワークスペース](https://wandb.ai/carey/table-test?workspace=user-carey)では、異なるバージョンからの行を同じテーブルに組み合わせる方法を示しています。

![](/images/data_vis/wandb_demo_toggle_on_and_off_cross_run_comparisons_in_tables.gif)

テーブルのフィルタ、ソート、およびグループ化機能を使用して、モデルの結果を探索および評価します。

![](/images/data_vis/wandb_demo_filter_on_a_table.png)