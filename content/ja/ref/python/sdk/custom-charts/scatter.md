---
title: scatter()
data_type_classification: function
menu:
  reference:
    identifier: ja-ref-python-sdk-custom-charts-scatter
object_type: python_sdk_custom_charts
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/plot/scatter.py >}}




### <kbd>関数</kbd> `scatter`

```python
scatter(
    table: 'wandb.Table',
    x: 'str',
    y: 'str',
    title: 'str' = '',
    split_table: 'bool' = False
) → CustomChart
```

wandb.Table のデータから散布図を作成します。 



**Args:**
 
 - `table`:  可視化するデータを含む W&B の Table。 
 - `x`:  x 軸に使用する列の名前。 
 - `y`:  y 軸に使用する列の名前。 
 - `title`:  散布図チャートのタイトル。 
 - `split_table`:  W&B の UI で テーブル を別セクションに分割するかどうか。`True` の場合、"Custom Chart Tables" という名前のセクションに表示されます。デフォルトは `False` です。 



**Returns:**
 
 - `CustomChart`:  W&B に ログ できるカスタム チャートのオブジェクト。チャートを ログ するには、`wandb.log()` に渡してください。 

**例:**
 ```python
import math
import random
import wandb

# 時間経過に伴う高度ごとの温度変動をシミュレート
data = [
    [i, random.uniform(-10, 20) - 0.005 * i + 5 * math.sin(i / 50)]
    for i in range(300)
]

# 高度 (m) と 温度 (°C) の列を持つ W&B の Table を作成
table = wandb.Table(data=data, columns=["altitude (m)", "temperature (°C)"])

# W&B の run を初期化し、散布図を ログ する
with wandb.init(project="temperature-altitude-scatter") as run:
    # 散布図を作成して ログ する
    scatter_plot = wandb.plot.scatter(
         table=table,
         x="altitude (m)",
         y="temperature (°C)",
         title="Altitude vs Temperature",
    )
    run.log({"altitude-temperature-scatter": scatter_plot})
```