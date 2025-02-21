---
title: Tables
description: データセットを反復してモデルの予測を理解する
cascade:
- url: guides/tables/:filename
menu:
  default:
    identifier: ja-guides-core-tables-_index
    parent: core
url: guides/tables
weight: 2
---

{{< cta-button productLink="https://wandb.ai/wandb/examples/reports/AlphaFold-ed-Proteins-in-W-B-Tables--Vmlldzo4ODc0MDc" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W%26B_Tables_Quickstart.ipynb" >}}

W&B Tables を使用して、表形式のデータを可視化およびクエリできます。例えば：

* 異なるモデルが同じテストセットでどのように動作するか比較する
* データのパターンを識別する
* モデルのサンプル予測を視覚的に確認する
* 一般的に誤分類された例を見つけるためにクエリを実行する

{{< img src="/images/data_vis/tables_sample_predictions.png" alt="" >}}

上の画像は、セマンティックセグメンテーションとカスタムメトリクスを含むテーブルを示しています。このテーブルは、ここでこの [W&B ML Course のサンプルプロジェクト](https://wandb.ai/av-team/mlops-course-001)で表示できます。

## 仕組み

Table は、各列が単一タイプのデータを持つ 2 次元のデータグリッドです。Tables は、プリミティブおよび数値型、ネストされたリスト、辞書、およびリッチメディア型をサポートします。

## テーブルをログに記録する

数行のコードでテーブルをログに記録します：

- [`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ja" >}}): 結果を追跡するための [run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を作成します。
- [`wandb.Table()`]({{< relref path="/ref/python/data-types/table.md" lang="ja" >}}): 新しいテーブルオブジェクトを作成します。
  - `columns`: 列名を設定します。
  - `data`: テーブルの内容を設定します。
- [`run.log()`]({{< relref path="/ref/python/log.md" lang="ja" >}}): テーブルをログに記録して W&B に保存します。

```python showLineNumbers
import wandb

run = wandb.init(project="table-test")
my_table = wandb.Table(columns=["a", "b"], data=[["a1", "b1"], ["a2", "b2"]])
run.log({"Table Name": my_table})
```

## 開始方法
* [クイックスタート]({{< relref path="./tables-walkthrough.md" lang="ja" >}}): データテーブルをログし、データを可視化し、データをクエリする方法を学びます。
* [Tables Gallery]({{< relref path="./tables-gallery.md" lang="ja" >}}): Tables のユースケース例を参照してください。