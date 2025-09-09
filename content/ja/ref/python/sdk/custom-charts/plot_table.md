---
title: plot_table()
data_type_classification: function
menu:
  reference:
    identifier: ja-ref-python-sdk-custom-charts-plot_table
object_type: python_sdk_custom_charts
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/plot/custom_chart.py >}}




### <kbd>関数</kbd> `plot_table`

```python
plot_table(
    vega_spec_name: 'str',
    data_table: 'wandb.Table',
    fields: 'dict[str, Any]',
    string_fields: 'dict[str, Any] | None' = None,
    split_table: 'bool' = False
) → CustomChart
```

Vega-Lite 仕様と `wandb.Table` を使ってカスタム チャートを作成します。

この関数は、Vega-Lite 仕様と、`wandb.Table` オブジェクトで表されるデータ テーブルに基づいてカスタム チャートを作成します。仕様はあらかじめ定義され、W&B バックエンドに保存されている必要があります。関数は、`wandb.Run.log()` を使って W&B にログできるカスタム チャート オブジェクトを返します。



**引数:**
 
 - `vega_spec_name`:  可視化の構造を定義する Vega-Lite 仕様の名前または識別子。 
 - `data_table`:  可視化するデータを含む `wandb.Table` オブジェクト。 
 - `fields`:  Vega-Lite 仕様のフィールドと、可視化するデータ テーブルの対応する列とのマッピング。 
 - `string_fields`:  カスタム可視化で必要な文字列定数に値を与えるための 辞書。 
 - `split_table`:  テーブルを W&B の UI で別セクションに分割するかどうか。`True` の場合、テーブルは "Custom Chart Tables" というセクションに表示されます。デフォルトは `False` です。 



**戻り値:**
 
 - `CustomChart`:  W&B にログできるカスタム チャート オブジェクト。チャートをログするには、チャート オブジェクトを `wandb.Run.log()` の引数として渡します。 



**例外:**
 
 - `wandb.Error`:  `data_table` が `wandb.Table` オブジェクトではない場合。 



**例:**
 ```python
# Vega-Lite 仕様とデータ テーブルを使ってカスタム チャートを作成します。
import wandb

data = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
table = wandb.Table(data=data, columns=["x", "y"])
fields = {"x": "x", "y": "y", "title": "MY TITLE"}

with wandb.init() as run:
    # トレーニング コードはここに書きます

    # `string_fields` でカスタム タイトルを作成します。
    my_custom_chart = wandb.plot_table(
         vega_spec_name="wandb/line/v0",
         data_table=table,
         fields=fields,
         string_fields={"title": "Title"},
    )

    run.log({"custom_chart": my_custom_chart})
```