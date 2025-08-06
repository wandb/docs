---
title: テーブル
description: データセットを繰り返し改善し、モデルの予測を理解する
menu:
  default:
    identifier: tables
    parent: models
weight: 2
url: guides/models/tables
cascade:
- url: guides/models/tables/:filename
---

{{< cta-button productLink="https://wandb.ai/wandb/examples/reports/AlphaFold-ed-Proteins-in-W-B-Tables--Vmlldzo4ODc0MDc" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W%26B_Tables_Quickstart.ipynb" >}}

W&B Tables を使って、表形式のデータを可視化・検索しましょう。例えば：

* 異なるモデルが同じテストセットでどうパフォーマンスするか比較する
* データ内のパターンを特定する
* モデルのサンプル予測をビジュアルで確認する
* よく誤分類されるサンプルをクエリで探す

{{< img src="/images/data_vis/tables_sample_predictions.png" alt="セマンティックセグメンテーション予測のテーブル" >}}
上の画像は、セマンティックセグメンテーションとカスタム メトリクスを含むテーブルの例です。このテーブルは [W&B ML コースのサンプルプロジェクト](https://wandb.ai/av-team/mlops-course-001) で閲覧できます。

## 仕組み

Table は、各カラムが1種類のデータ型を持つ、2次元グリッド状のデータです。Tables では、プリミティブ型や数値型だけでなく、ネストしたリストや辞書、リッチメディア型もサポートしています。

## Table をログする

数行のコードでテーブルをログできます：

- [`wandb.init()`]({{< relref "/ref/python/sdk/functions/init.md" >}}): 結果追跡用の [run]({{< relref "/guides/models/track/runs/" >}}) を作成します。
- [`wandb.Table()`]({{< relref "/ref/python/sdk/data-types/table.md" >}}): 新しいテーブル オブジェクトを作成します。
  - `columns`: カラム名を設定します。
  - `data`: テーブルの内容を設定します。
- [`run.log()`]({{< relref "/ref/python/sdk/classes/run.md/#method-runlog" >}}): テーブルを W&B に保存します。

```python
import wandb

run = wandb.init(project="table-test")
my_table = wandb.Table(columns=["a", "b"], data=[["a1", "b1"], ["a2", "b2"]])
run.log({"Table Name": my_table})
```

## 開始方法
* [クイックスタート]({{< relref "./tables-walkthrough.md" >}}): データテーブルのログ・データの可視化・データクエリの方法を学べます。
* [Tables ギャラリー]({{< relref "./tables-gallery.md" >}}): Tables のユースケース例を紹介しています。