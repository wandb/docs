---
title: line()
object_type: python_sdk_custom_charts
data_type_classification: function
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/plot/line.py >}}




### <kbd>関数</kbd> `line`

```python
line(
    table: 'wandb.Table',
    x: 'str',
    y: 'str',
    stroke: 'str | None' = None,
    title: 'str' = '',
    split_table: 'bool' = False
) → CustomChart
```

カスタマイズ可能なラインチャートを作成します。



**引数:**

 - `table`:   チャート用のデータが格納されたテーブル。
 - `x`:  x軸の値に使用するカラム名。
 - `y`:  y軸の値に使用するカラム名。
 - `stroke`:  線の種類を区別するためのカラム名（例: ラインのグループ化など）。
 - `title`:  チャートのタイトル。
 - `split_table`:  テーブルを W&B UI 内で別セクションに分割して表示するかどうか。`True` の場合、「Custom Chart Tables」というセクションに表示されます。デフォルトは `False` です。



**戻り値:**

 - `CustomChart`:  W&B にログできるカスタムチャート オブジェクト。チャートをログするには `wandb.log()` に渡してください。



**例:**

```python
import math
import random
import wandb

# 異なるパターンの複数のデータ系列を作成
data = []
for i in range(100):
     # シリーズ1: ランダムノイズ付き正弦波
     data.append([i, math.sin(i / 10) + random.uniform(-0.1, 0.1), "series_1"])
     # シリーズ2: ランダムノイズ付き余弦波
     data.append([i, math.cos(i / 10) + random.uniform(-0.1, 0.1), "series_2"])
     # シリーズ3: ランダムノイズ付き線形増加
     data.append([i, i / 10 + random.uniform(-0.5, 0.5), "series_3"])

# テーブルのカラムを定義
table = wandb.Table(data=data, columns=["step", "value", "series"])

# wandb run を初期化し、ラインチャートをログ
with wandb.init(project="line_chart_example") as run:
     line_chart = wandb.plot.line(
         table=table,
         x="step",
         y="value",
         stroke="series",  # 「series」カラムでグループ化
         title="Multi-Series Line Plot",
     )
     run.log({"line-chart": line_chart})
```