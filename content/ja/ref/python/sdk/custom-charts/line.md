---
title: line()
data_type_classification: function
menu:
  reference:
    identifier: ja-ref-python-sdk-custom-charts-line
object_type: python_sdk_custom_charts
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

カスタマイズ可能な折れ線グラフを作成します。 



**引数:**
 
 - `table`:   チャート用のデータを含むテーブル。 
 - `x`:  x 軸の値の列名。 
 - `y`:  y 軸の値の列名。 
 - `stroke`:  線を区別するための列名（例: 線のグループ化）。 
 - `title`:  チャートのタイトル。 
 - `split_table`:  W&B の UI でテーブルを別セクションに分けて表示するかどうか。`True` の場合、"Custom Chart Tables" というセクションに表示されます。デフォルトは `False`。 



**戻り値:**
 
 - `CustomChart`:  W&B にログできるカスタムチャートのオブジェクト。チャートをログするには、`wandb.log()` に渡します。 



**例:**
 

```python
import math
import random
import wandb

# 異なるパターンの複数の系列データを作成
data = []
for i in range(100):
     # 系列 1: ランダムなノイズを加えた正弦波パターン
     data.append([i, math.sin(i / 10) + random.uniform(-0.1, 0.1), "series_1"])
     # 系列 2: ランダムなノイズを加えた余弦波パターン
     data.append([i, math.cos(i / 10) + random.uniform(-0.1, 0.1), "series_2"])
     # 系列 3: ランダムなノイズを加えた線形増加
     data.append([i, i / 10 + random.uniform(-0.5, 0.5), "series_3"])

# テーブルの列を定義
table = wandb.Table(data=data, columns=["step", "value", "series"])

# wandb run を初期化し、折れ線グラフをログする
with wandb.init(project="line_chart_example") as run:
     line_chart = wandb.plot.line(
         table=table,
         x="step",
         y="value",
         stroke="series",  # 「series」列でグループ化
         title="Multi-Series Line Plot",
     )
     run.log({"line-chart": line_chart})
```