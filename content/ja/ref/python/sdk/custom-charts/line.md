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
 
 - `table`:   グラフ用のデータを含むテーブル。 
 - `x`:  x軸となる値のカラム名。 
 - `y`:  y軸となる値のカラム名。 
 - `stroke`:  線のグループ分けなどに使うカラム名。 
 - `title`:  グラフのタイトル。 
 - `split_table`:  テーブルを W&B UI の別セクションに分割表示するかどうか。`True` の場合、「Custom Chart Tables」というセクションに表示されます。デフォルトは `False` です。



**戻り値:**
 
 - `CustomChart`:  W&B にログ可能なカスタムチャートオブジェクト。チャートをログする際は `wandb.log()` に渡してください。



**使用例:**
 

```python
import math
import random
import wandb

# 異なるパターンで複数の系列のデータを作成
data = []
for i in range(100):
     # シリーズ 1: サイン波 + ランダムノイズ
     data.append([i, math.sin(i / 10) + random.uniform(-0.1, 0.1), "series_1"])
     # シリーズ 2: コサイン波 + ランダムノイズ
     data.append([i, math.cos(i / 10) + random.uniform(-0.1, 0.1), "series_2"])
     # シリーズ 3: 線形増加 + ランダムノイズ
     data.append([i, i / 10 + random.uniform(-0.5, 0.5), "series_3"])

# テーブルのカラムを定義
table = wandb.Table(data=data, columns=["step", "value", "series"])

# wandb run を初期化し、折れ線グラフをログ
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