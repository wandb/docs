---
title: テーブル
description: データセットを反復し、モデルの予測を理解する
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

W&B Tables を使って、表形式データの可視化やクエリができます。例えば:

* 異なるモデルが同じテストセットでどのように動作するか比較する
* データ内のパターンを見つける
* モデルのサンプル予測結果を直感的に確認する
* よく誤分類される例をクエリして発見する

{{< img src="/images/data_vis/tables_sample_predictions.png" alt="セマンティックセグメンテーション予測のテーブル" >}}
上記の画像は、セマンティックセグメンテーションとカスタムメトリクスが表示されたテーブル例です。このテーブルは [W&B ML Course のサンプルプロジェクト](https://wandb.ai/av-team/mlops-course-001) で見ることができます。

## 仕組み

Table は、各列が一つの型を持つ 2 次元グリッドのデータです。Tables では、基本型や数値型だけでなく、ネストされたリストや辞書、リッチメディア型もサポートしています。

## Table をログする

数行のコードで Table をログできます:

- [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}): 結果を記録する [run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を作成します。
- [`wandb.Table()`]({{< relref path="/ref/python/sdk/data-types/table.md" lang="ja" >}}): 新しいテーブルオブジェクトを作成します。
  - `columns`: 列名を設定します。
  - `data`: テーブルの内容を設定します。
- [`run.log()`]({{< relref path="/ref/python/sdk/classes/run.md/#method-runlog" lang="ja" >}}): テーブルを W&B に保存します。

```python
import wandb

run = wandb.init(project="table-test")
my_table = wandb.Table(columns=["a", "b"], data=[["a1", "b1"], ["a2", "b2"]])
run.log({"Table Name": my_table})
```

## 開始方法
* [クイックスタート]({{< relref path="./tables-walkthrough.md" lang="ja" >}}): データテーブルのログ、可視化、クエリ方法を学びます。
* [Tables ギャラリー]({{< relref path="./tables-gallery.md" lang="ja" >}}): Tables のユースケース例をチェックできます。