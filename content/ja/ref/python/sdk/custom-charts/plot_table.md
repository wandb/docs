---
title: 'plot_table()

  '
object_type: python_sdk_custom_charts
data_type_classification: function
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

この関数は、Vega-Lite 仕様と `wandb.Table` オブジェクトとして表現されたデータテーブルに基づいてカスタムチャートを作成します。仕様はあらかじめ定義され、W&B バックエンドに保存されている必要があります。この関数は、`wandb.Run.log()` を使って W&B にログできるカスタムチャートオブジェクトを返します。



**引数:**
 
 - `vega_spec_name`:  可視化の構造を定義する Vega-Lite 仕様の名前または識別子。
 - `data_table`:  可視化されるデータを含む `wandb.Table` オブジェクト。
 - `fields`:  Vega-Lite 仕様のフィールドと、可視化するデータテーブル内の対応するカラムをマッピングします。
 - `string_fields`:  カスタム可視化に必要な文字列定数に値を指定するための辞書。
 - `split_table`:  テーブルを W&B UI 内で別セクションとして分割するかどうか。`True` の場合、「Custom Chart Tables」というセクションにテーブルが表示されます。デフォルトは `False`。



**戻り値:**
 
 - `CustomChart`:  W&B にログできるカスタムチャートオブジェクト。チャートをログするには、このオブジェクトを `wandb.Run.log()` の引数として渡してください。



**例外:**
 
 - `wandb.Error`:  `data_table` が `wandb.Table` オブジェクトでない場合に発生します。



**例:**
 ```python
# Vega-Lite 仕様とデータテーブルを使ってカスタムチャートを作成します。
import wandb

data = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
table = wandb.Table(data=data, columns=["x", "y"])
fields = {"x": "x", "y": "y", "title": "MY TITLE"}

with wandb.init() as run:
    # トレーニングのコードをここに記述

    # `string_fields` を使ってカスタムタイトルを作成
    my_custom_chart = wandb.plot_table(
         vega_spec_name="wandb/line/v0",
         data_table=table,
         fields=fields,
         string_fields={"title": "Title"},
    )

    run.log({"custom_chart": my_custom_chart})
```