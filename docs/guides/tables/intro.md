---
description: "\u30C7\u30FC\u30BF\u30BB\u30C3\u30C8\u3092\u53CD\u5FA9\u3057\u3001\u30E2\
  \u30C7\u30EB\u306E\u4E88\u6E2C\u3092\u7406\u89E3\u3059\u308B"
slug: /guides/tables
displayed_sidebar: default
---
import Translate, {translate} from '@docusaurus/Translate';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# データを可視化する

<CTAButtons productLink="https://wandb.ai/wandb/examples/reports/AlphaFold-ed-Proteins-in-W-B-Tables--Vmlldzo4ODc0MDc" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W%26B_Tables_Quickstart.ipynb"/>

W&B Tablesを使用して、表形式のデータを可視化およびクエリします。例えば：

* 異なるモデルが同じテストセットでどのようにパフォーマンスを発揮するかを比較
* データ内のパターンを識別
* モデルの予測サンプルを視覚的に確認
* よく誤分類される例をクエリで検索

![](/images/data_vis/tables_sample_predictions.png)
上の画像は、セマンティックセグメンテーションとカスタムメトリクスを含むテーブルを示しています。このテーブルは、W&B ML Courseの[サンプルプロジェクト](https://wandb.ai/av-team/mlops-course-001)で確認できます。

## 仕組み

Tableは、各列が単一のデータ型を持つ二次元のデータグリッドです。Tablesは、プリミティブ型や数値型、ネストされたリスト、辞書、リッチメディア型をサポートします。

## Tableをログする

数行のコードでTableをログします：

- [`wandb.init()`](../../ref/python/init.md): 結果を追跡するための[run](../runs/intro.md)を作成します。
- [`wandb.Table()`](../../ref/python/data-types/table.md): 新しいテーブルオブジェクトを作成します。
  - `columns`: 列名を設定します。
  - `data`: テーブルの内容を設定します。
- [`run.log()`](../../ref/python/log.md): テーブルをログしてW&Bに保存します。

```python showLineNumbers
import wandb

run = wandb.init(project="table-test")
my_table = wandb.Table(columns=["a", "b"], data=[["a1", "b1"], ["a2", "b2"]])
run.log({"Table Name": my_table})
```

## 開始方法
* [クイックスタート](./tables-walkthrough.md): データテーブルのログ、データの可視化、データのクエリ方法を学びます。
* [Tables Gallery](./tables-gallery.md): Tablesのユースケース例を参照します。