---
title: テーブル
description: データセットを繰り返し、モデルの予測を理解する
cascade:
- url: /ja/guides/models/tables/:filename
menu:
  default:
    identifier: ja-guides-models-tables-_index
    parent: models
url: /ja/guides/models/tables
weight: 2
---

{{< cta-button productLink="https://wandb.ai/wandb/examples/reports/AlphaFold-ed-Proteins-in-W-B-Tables--Vmlldzo4ODc0MDc" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W%26B_Tables_Quickstart.ipynb" >}}

W&B Tables を使用して、表形式のデータを視覚化し、クエリを実行します。 例:

* 同じテストセットで異なるモデルがどのように機能するかを比較する
* データのパターンを特定する
* サンプルモデルの予測を視覚的に確認する
* よく誤分類される例を見つけるためにクエリを実行する

{{< img src="/images/data_vis/tables_sample_predictions.png" alt="" >}}
上の画像は、セマンティックセグメンテーションとカスタムメトリクスを含むテーブルを示しています。このテーブルはこちらの [W&B ML Course のサンプルプロジェクト](https://wandb.ai/av-team/mlops-course-001) で見ることができます。

## 仕組み

Table は、各列が単一のデータ型を持つデータの 2 次元グリッドです。Tables はプリミティブ型と数値型、さらに入れ子リスト、辞書、およびリッチメディア型をサポートします。

## Table をログする

数行のコードでテーブルをログします:

- [`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ja" >}}):  [run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を作成して結果を追跡します。
- [`wandb.Table()`]({{< relref path="/ref/python/data-types/table.md" lang="ja" >}}): 新しいテーブルオブジェクトを作成します。
  - `columns`: 列の名前を設定します。
  - `data`: テーブルの内容を設定します。
- [`run.log()`]({{< relref path="/ref/python/log.md" lang="ja" >}}): テーブルをログして W&B に保存します。

```python
import wandb

run = wandb.init(project="table-test")
my_table = wandb.Table(columns=["a", "b"], data=[["a1", "b1"], ["a2", "b2"]])
run.log({"Table Name": my_table})
```

## 開始方法
* [クイックスタート]({{< relref path="./tables-walkthrough.md" lang="ja" >}}): データテーブルのログ、データの視覚化、データのクエリについて学びます。
* [Tables Gallery]({{< relref path="./tables-gallery.md" lang="ja" >}}): Tables のユースケース例を参照します。