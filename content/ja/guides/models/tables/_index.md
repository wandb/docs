---
title: Tables
description: データセットを反復処理し、モデル の予測を理解する
cascade:
- url: guides/models/tables/:filename
menu:
  default:
    identifier: ja-guides-models-tables-_index
    parent: models
url: guides/models/tables
weight: 2
---

{{< cta-button productLink="https://wandb.ai/wandb/examples/reports/AlphaFold-ed-Proteins-in-W-B-Tables--Vmlldzo4ODc0MDc" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W%26B_Tables_Quickstart.ipynb" >}}

W&B Tables を使用して、テーブル形式のデータを可視化およびクエリします。例:

* 同じテストセットで異なるモデルがどのように機能するかを比較する
* データ内のパターンを特定する
* サンプルモデルの予測を視覚的に確認する
* 一般的に誤分類された例を見つけるためにクエリする

{{< img src="/images/data_vis/tables_sample_predictions.png" alt="" >}}
上の画像は、セマンティックセグメンテーションとカスタムメトリクスを含むテーブルを示しています。このテーブルは、[W&B ML コースのサンプル project](https://wandb.ai/av-team/mlops-course-001) でご覧ください。

## 仕組み

Table は、各カラムが単一のデータ型を持つデータの二次元グリッドです。Tables は、プリミティブ型と数値型、およびネストされたリスト、辞書、リッチメディア型をサポートしています。

## Table をログする

数行のコードで Table をログします。

- [`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ja" >}}): [run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を作成して、結果を追跡します。
- [`wandb.Table()`]({{< relref path="/ref/python/data-types/table.md" lang="ja" >}}): 新しい table オブジェクトを作成します。
  - `columns`: カラム名を設定します。
  - `data`: テーブルの内容を設定します。
- [`run.log()`]({{< relref path="/ref/python/log.md" lang="ja" >}}): テーブルをログして、W&B に保存します。

```python
import wandb

run = wandb.init(project="table-test")
my_table = wandb.Table(columns=["a", "b"], data=[["a1", "b1"], ["a2", "b2"]])
run.log({"Table Name": my_table})
```

## 開始方法
* [クイックスタート]({{< relref path="./tables-walkthrough.md" lang="ja" >}}): データテーブルのログ、データの可視化、およびデータのクエリを学習します。
* [Tables Gallery]({{< relref path="./tables-gallery.md" lang="ja" >}}): Tables のユースケースの例をご覧ください。
