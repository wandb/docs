---
title: 'scatter()

  '
data_type_classification: function
menu:
  reference:
    identifier: ko-ref-python-sdk-custom-charts-scatter
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

wandb.Table의 데이터를 사용해 산점도를 만듭니다.



**Args:**
 
 - `table`:  시각화할 데이터를 포함하고 있는 W&B Table 입니다.
 - `x`:  x축에 사용할 컬럼 이름입니다.
 - `y`:  y축에 사용할 컬럼 이름입니다.
 - `title`:  산점도의 제목입니다.
 - `split_table`:  테이블을 W&B UI에서 별도의 섹션으로 분리해서 보여줄지 여부입니다. `True`로 설정하면 "Custom Chart Tables"라는 섹션에 테이블이 표시됩니다. 기본값은 `False`입니다.



**Returns:**
 
 - `CustomChart`:  W&B에 로그할 수 있는 커스텀 차트 오브젝트입니다. 차트를 로그하려면 `wandb.log()`에 전달하세요.

**Example:**
 ```python
import math
import random
import wandb

# 다양한 고도에서 시간에 따른 온도 변화를 시뮬레이션합니다.
data = [
    [i, random.uniform(-10, 20) - 0.005 * i + 5 * math.sin(i / 50)]
    for i in range(300)
]

# 고도(m)와 온도(°C) 컬럼을 가진 W&B Table을 만듭니다.
table = wandb.Table(data=data, columns=["altitude (m)", "temperature (°C)"])

# W&B run을 초기화하고 산점도 로그하기
with wandb.init(project="temperature-altitude-scatter") as run:
    # 산점도 생성 및 로그
    scatter_plot = wandb.plot.scatter(
         table=table,
         x="altitude (m)",
         y="temperature (°C)",
         title="Altitude vs Temperature",
    )
    run.log({"altitude-temperature-scatter": scatter_plot})
```