---
description: データセットを繰り返し、モデルの予測を理解する
slug: /guides/tables
displayed_sidebar: default
---

import Translate, {translate} from '@docusaurus/Translate';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';


# データの可視化

<CTAButtons productLink="https://wandb.ai/wandb/examples/reports/AlphaFold-ed-Proteins-in-W-B-Tables--Vmlldzo4ODc0MDc" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W%26B_Tables_Quickstart.ipynb"/>

W&B Tables を使って表形式のデータを可視化およびクエリします。例えば：

* 異なるモデルが同じテストセットでどのようにパフォーマンスを発揮するか比較する
* データのパターンを特定する
* モデルの予測サンプルを視覚的に確認する
* 誤分類されやすい例をクエリで見つける

![](/images/data_vis/tables_sample_predictions.png)
上の画像はセマンティックセグメンテーションとカスタムメトリクスを含むテーブルを示しています。このテーブルは [W&B ML Course のサンプルプロジェクト](https://wandb.ai/av-team/mlops-course-001)で確認できます。

## 仕組み

Tableは、各列が単一のデータ型を持つ2次元のデータグリッドです。Tablesは、プリミティブ型や数値型、入れ子リスト、辞書、リッチメディア型をサポートします。

## Table をログする

数行のコードでテーブルをログします：

- [`wandb.init()`](../../ref/python/init.md): 結果を追跡するための[run](../runs/intro.md)を作成します。
- [`wandb.Table()`](../../ref/python/data-types/table.md): 新しいテーブルオブジェクトを作成します。
  - `columns`: 列の名前を設定します。
  - `data`: テーブルの内容を設定します。
- [`run.log()`](../../ref/python/log.md): テーブルをログして W&B に保存します。

```python showLineNumbers
import wandb

run = wandb.init(project="table-test")
my_table = wandb.Table(columns=["a", "b"], data=[["a1", "b1"], ["a2", "b2"]])
run.log({"Table Name": my_table})
```

## 開始方法
* [クイックスタート](./tables-walkthrough.md): データテーブルのログ、データの可視化、データのクエリ方法を学びます。
* [Tables Gallery](./tables-gallery.md): Tables のユースケースの例を参照します。