---
title: histogram()
data_type_classification: function
menu:
  reference:
    identifier: ko-ref-python-sdk-custom-charts-Histogram
object_type: python_sdk_custom_charts
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/plot/histogram.py >}}




### <kbd>function</kbd> `histogram`

```python
histogram(
    table: 'wandb.Table',
    value: 'str',
    title: 'str' = '',
    split_table: 'bool' = False
) → CustomChart
```

W&B Table 에서 히스토그램 차트를 생성합니다.



**인자:**
 
 - `table`:  히스토그램에 사용할 데이터를 포함한 W&B Table 입니다.
 - `value`:  bin 축(x축)에 대한 라벨입니다.
 - `title`:  히스토그램 플롯의 제목입니다.
 - `split_table`:  이 Table 을 W&B UI 내에서 별도의 섹션으로 분리해서 보여줄지 여부입니다. `True`로 설정하면, 해당 Table 은 "Custom Chart Tables"라는 섹션에 표시됩니다. 기본값은 `False`입니다.



**반환값:**
 
 - `CustomChart`:  W&B에 로그할 수 있는 커스텀 차트 오브젝트입니다. 차트를 로그하려면 `wandb.log()`에 전달하세요.




**예시:**
 

```python
import math
import random
import wandb

# 랜덤 데이터 생성
data = [[i, random.random() + math.sin(i / 10)] for i in range(100)]

# W&B Table 생성
table = wandb.Table(
    data=data,
    columns=["step", "height"],
)

# 히스토그램 플롯 생성
histogram = wandb.plot.histogram(
    table,
    value="height",
    title="My Histogram",
)

# 히스토그램 플롯을 W&B에 로그
with wandb.init(...) as run:
    run.log({"histogram-plot1": histogram})
```