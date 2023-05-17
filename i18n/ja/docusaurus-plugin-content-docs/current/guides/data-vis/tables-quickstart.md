---
description: Explore how to use W&B Tables with this 5 minute Quickstart.
displayed_sidebar: default
---

# クイックスタート

以下のクイックスタートでは、データテーブルのロギング、データの可視化、データのクエリ方法を示します。

下のボタンを選択して、MNISTデータを使用したPyTorchのクイックスタート例プロジェクトを試してください。 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://wandb.me/tables-quickstart)

## 1. テーブルをログに記録する

W&Bでテーブルをログに記録するための手順は以下の通りです。
1. [`wandb.init()`](../../ref/python/init.md) で W&B Runを初期化します。
2. [`wandb.Table()`](../../ref/python/data-types/table.md)オブジェクトのインスタンスを作成します。`columns` と `data` パラメータに、それぞれテーブルの列名とデータを渡します。
3. [`run.log()`](../../ref/python/log.md) を使ってテーブルをキーと値のペアとしてログに記録します。キーにはテーブルの名前を指定し、値には `wandb.Table` のオブジェクトインスタンスを渡します。

```python
run = wandb.init(project="table-test")
my_table = wandb.Table(
    columns=["a", "b"], 
    data=[["a1", "b1"], ["a2", "b2"]]
    )
run.log({"Table Name": my_table})   
```

オプションで、PandasのDataFrameを `wandb.Table()` クラスに渡すことができます。対応しているデータ型の詳細については、W&B APIリファレンスガイドの [`wandb.Table`](../../ref/python/data-types/table.md) を参照してください。

## 2. ワークスペースでテーブルを可視化する
ワークスペースで結果の表を確認してください。W&Bアプリに移動し、プロジェクトのワークスペースでRunの名前を選択します。固有の表キーごとに新しいパネルが追加されます。

![](/images/data_vis/wandb_demo_logged_sample_table.png)

この例では、`my_table`がキー`"Table Name"`のもとに記録されています。

## 3. モデルバージョン間の比較

複数のW&B Runからサンプルテーブルを記録し、プロジェクトワークスペースで結果を比較します。この[例のワークスペース](https://wandb.ai/carey/table-test?workspace=user-carey)では、同じテーブルで複数の異なるバージョンの行を組み合わせる方法を示しています。

![](/images/data_vis/wandb_demo_toggle_on_and_off_cross_run_comparisons_in_tables.gif)

テーブルのフィルター、ソート、グループ化機能を使用して、モデルの結果を調査し、評価します。

![](/images/data_vis/wandb_demo_filter_on_a_table.png)