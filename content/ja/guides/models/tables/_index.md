---
title: テーブル
description: データセットを反復的に改善し、モデルの予測を理解する
aliases:
- /guides/models/tables/
cascade:
- url: guides/tables/:filename
menu:
  default:
    identifier: ja-guides-models-tables-_index
    parent: models
url: guides/tables
weight: 2
---

{{< cta-button productLink="https://wandb.ai/wandb/examples/reports/AlphaFold-ed-Proteins-in-W-B-Tables--Vmlldzo4ODc0MDc" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W%26B_Tables_Quickstart.ipynb" >}}

W&B Tables を使って表形式のデータを可視化・クエリできます。例えば:

* 同じ テストセット で異なるモデルの性能を比較
* データ内のパターンを特定
* サンプルのモデル予測を視覚的に確認
* よく誤分類されるサンプルを見つけるためにクエリを実行

{{< img src="/images/data_vis/tables_sample_predictions.png" alt="セマンティック セグメンテーションの予測テーブル" >}}
上の画像は、セマンティック セグメンテーションとカスタム メトリクスを含むテーブルを示しています。W&B の ML コースのこの [サンプル Project](https://wandb.ai/av-team/mlops-course-001) で、このテーブルを閲覧できます。

## 仕組み

Table は、各列が 1 種類のデータ型を持つ 2 次元のデータグリッドです。Table は、プリミティブ型や数値型に加えて、ネストしたリスト、辞書、リッチメディア型をサポートします。 

## Table をログする

数行のコードで Table をログします:

- [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}): 結果を追跡するための [run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を作成します。
- [`wandb.Table()`]({{< relref path="/ref/python/sdk/data-types/table.md" lang="ja" >}}): 新しい Table オブジェクトを作成します。
  - `columns`: 列名を設定します。
  - `data`: テーブルの内容を設定します。
- [`run.log()`]({{< relref path="/ref/python/sdk/classes/run.md/#method-runlog" lang="ja" >}}): Table をログして W&B に保存します。

```python
import wandb

run = wandb.init(project="table-test")
my_table = wandb.Table(columns=["a", "b"], data=[["a1", "b1"], ["a2", "b2"]])
run.log({"Table Name": my_table})
```

## 開始方法
* [クイックスタート]({{< relref path="./tables-walkthrough.md" lang="ja" >}}): データテーブルをログし、データを可視化し、データにクエリする方法を学びます。
* [Tables ギャラリー]({{< relref path="./tables-gallery.md" lang="ja" >}}): Tables のユースケース例を確認できます。