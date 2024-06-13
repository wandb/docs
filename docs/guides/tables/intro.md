---
description: データセットを反復し、モデルの予測を理解する
slug: /guides/tables
displayed_sidebar: default
---
import Translate, {translate} from '@docusaurus/Translate';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# データの可視化

<CTAButtons productLink="https://wandb.ai/wandb/examples/reports/AlphaFold-ed-Proteins-in-W-B-Tables--Vmlldzo4ODc0MDc" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W%26B_Tables_Quickstart.ipynb"/>

W&B Tablesを使用して、表形式のデータを可視化およびクエリします。例えば：

* 異なるモデルが同じテストセットでどのようにパフォーマンスするかを比較
* データ内のパターンを特定
* サンプルモデルの予測を視覚的に確認
* よく誤分類される例をクエリで検索

![](/images/data_vis/tables_sample_predictions.png)
上の画像は、セマンティックセグメンテーションとカスタムメトリクスを含むテーブルを示しています。このテーブルは、W&B MLコースの[サンプルプロジェクト](https://wandb.ai/av-team/mlops-course-001)で確認できます。

## 仕組み

Tableは、各列が単一のデータ型を持つ2次元のデータグリッドです。Tablesは、プリミティブ型や数値型、ネストされたリスト、辞書、リッチメディア型をサポートしています。

## Tableをログする

数行のコードでTableをログします：

- [`wandb.init()`](../../ref/python/init.md): 結果を追跡するための[run](../runs/intro.md)を作成。
- [`wandb.Table()`](../../ref/python/data-types/table.md): 新しいTableオブジェクトを作成。
  - `columns`: 列名を設定。
  - `data`: テーブルの内容を設定。
- [`run.log()`](../../ref/python/log.md): TableをログしてW&Bに保存。

```python showLineNumbers
import wandb

run = wandb.init(project="table-test")
my_table = wandb.Table(columns=["a", "b"], data=[["a1", "b1"], ["a2", "b2"]])
run.log({"Table Name": my_table})
```

## 開始方法
* [クイックスタート](./tables-walkthrough.md): データテーブルのログ、データの可視化、データのクエリを学びます。
* [Tables Gallery](./tables-gallery.md): Tablesのユースケースの例を参照します。