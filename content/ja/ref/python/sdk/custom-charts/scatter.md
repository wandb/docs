---
title: scatter()
data_type_classification: function
menu:
  reference:
    identifier: ja-ref-python-sdk-custom-charts-scatter
object_type: python_sdk_custom_charts
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/plot/scatter.py >}}

### <kbd>function</kbd> `scatter`

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

**引数:**

 - `table`:  可視化したいデータを含む W&B Table。
 - `x`:  x 軸に使用するカラム名。
 - `y`:  y 軸に使用するカラム名。
 - `title`:  散布図のタイトル。
 - `split_table`:  W&B UI でテーブルを別セクションに分けて表示するかどうか。 `True` の場合、「Custom Chart Tables」というセクションで表示されます。デフォルトは `False` です。

**戻り値:**

 - `CustomChart`:  W&B にログできるカスタムチャートオブジェクト。チャートをログするには `wandb.log()` に渡します。

**使用例:**
 ```python
import math
import random
import wandb

# 時間経過による異なる高度での気温変動をシミュレーション
data = [
    [i, random.uniform(-10, 20) - 0.005 * i + 5 * math.sin(i / 50)]
    for i in range(300)
]

# 高度（m）と気温（°C）のカラムを持つ W&B table を作成
table = wandb.Table(data=data, columns=["altitude (m)", "temperature (°C)"])

# W&B run を初期化し、散布図をログ
with wandb.init(project="temperature-altitude-scatter") as run:
    # 散布図を作成してログ
    scatter_plot = wandb.plot.scatter(
         table=table,
         x="altitude (m)",
         y="temperature (°C)",
         title="Altitude vs Temperature",
    )
    run.log({"altitude-temperature-scatter": scatter_plot})
```