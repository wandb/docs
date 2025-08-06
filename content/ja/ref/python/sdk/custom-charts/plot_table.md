---
title: 'plot_table()

  '
data_type_classification: function
menu:
  reference:
    identifier: ja-ref-python-sdk-custom-charts-plot_table
object_type: python_sdk_custom_charts
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/plot/custom_chart.py >}}




### <kbd>function</kbd> `plot_table`

```python
plot_table(
    vega_spec_name: 'str',
    data_table: 'wandb.Table',
    fields: 'dict[str, Any]',
    string_fields: 'dict[str, Any] | None' = None,
    split_table: 'bool' = False
) → CustomChart
```

Vega-Lite の仕様と `wandb.Table` を使ってカスタムチャートを作成します。

この関数は、Vega-Lite 仕様と `wandb.Table` オブジェクトで表されるデータテーブルにもとづいてカスタムチャートを作成します。仕様はあらかじめ定義され、W&B バックエンドに保存されている必要があります。関数はカスタムチャートオブジェクトを返し、これは `wandb.Run.log()` を使って W&B にログできます。



**引数:**
 
 - `vega_spec_name`:  ビジュアライゼーションの構造を定義する Vega-Lite 仕様の名前または識別子。
 - `data_table`:  可視化したいデータを含んだ `wandb.Table` オブジェクト。
 - `fields`:  Vega-Lite 仕様中のフィールドと可視化対象となるデータテーブル内のカラムとを対応づけるマッピング。
 - `string_fields`:  カスタム可視化で必要な文字列定数に値をセットするための辞書。
 - `split_table`:  テーブルを W&B UI 内の独立したセクションに分割して表示するかどうか。`True` の場合、「Custom Chart Tables」というセクションで表示されます。デフォルトは `False`。



**戻り値:**
 
 - `CustomChart`:  W&B にログできるカスタムチャートオブジェクト。チャートをログするには、`wandb.Run.log()` への引数として渡してください。



**例外:**
 
 - `wandb.Error`:  `data_table` が `wandb.Table` オブジェクトでない場合に発生します。



**例:**
```python
# Vega-Lite の仕様とデータテーブルを使ってカスタムチャートを作成
import wandb

data = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
table = wandb.Table(data=data, columns=["x", "y"])
fields = {"x": "x", "y": "y", "title": "MY TITLE"}

with wandb.init() as run:
    # トレーニングコードはここに記述

    # `string_fields` でカスタムタイトルを設定
    my_custom_chart = wandb.plot_table(
         vega_spec_name="wandb/line/v0",
         data_table=table,
         fields=fields,
         string_fields={"title": "Title"},
    )

    run.log({"custom_chart": my_custom_chart})
```