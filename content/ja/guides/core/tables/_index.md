---
title: Tables
description: データセットを反復処理し、モデルの予測を理解する
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

W&B Tables を使用して、表形式のデータを可視化し、クエリを実行します。例:

* 同じ テストセット での異なる モデル のパフォーマンスを比較する
* データ のパターンを特定する
* サンプル モデル の 予測 を視覚的に確認する
* 一般的に誤分類される例を検索する クエリ


{{< img src="/images/data_vis/tables_sample_predictions.png" alt="" >}}
上の画像は、 セマンティックセグメンテーション とカスタム メトリクス を含む テーブル を示しています。この テーブル は、[W&B ML course のサンプル プロジェクト](https://wandb.ai/av-team/mlops-course-001)でご覧ください。

## 仕組み

Table は、各カラムが単一のデータ型を持つ、2 次元データ グリッドです。Table は、プリミティブ型と数値型、およびネストされたリスト、辞書、リッチ メディア型をサポートしています。

## Table を ログ する

数行の コード で テーブル を ログ します。

- [`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ja" >}}): 結果を追跡するための [run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を作成します。
- [`wandb.Table()`]({{< relref path="/ref/python/data-types/table.md" lang="ja" >}}): 新しい テーブル オブジェクト を作成します。
  - `columns`: カラム名を設定します。
  - `data`: テーブル のコンテンツを設定します。
- [`run.log()`]({{< relref path="/ref/python/log.md" lang="ja" >}}): テーブル を ログ に記録して、W&B に保存します。

```python showLineNumbers
import wandb

run = wandb.init(project="table-test")
my_table = wandb.Table(columns=["a", "b"], data=[["a1", "b1"], ["a2", "b2"]])
run.log({"Table Name": my_table})
```

## 開始方法
* [クイックスタート]({{< relref path="./tables-walkthrough.md" lang="ja" >}}): データ テーブル の ログ 、データの可視化、およびデータの クエリ を学習します。
* [Tables Gallery]({{< relref path="./tables-gallery.md" lang="ja" >}}): Tables の ユースケース の例をご覧ください。
