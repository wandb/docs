---
title: "line()  \n"
data_type_classification: function
menu:
  reference:
    identifier: ko-ref-python-sdk-custom-charts-line
object_type: python_sdk_custom_charts
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/plot/line.py >}}




### <kbd>function</kbd> `line`

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

사용자가 원하는 대로 커스터마이즈할 수 있는 선(line) 차트를 생성합니다. 



**인자:**
 
 - `table`:   차트에 사용할 데이터를 담고 있는 테이블입니다. 
 - `x`:  x축 값에 사용할 컬럼 이름입니다. 
 - `y`:  y축 값에 사용할 컬럼 이름입니다. 
 - `stroke`:  선을 구분하기 위한 컬럼 이름(예: 여러 그룹의 선 표현). 
 - `title`:  차트의 제목입니다. 
 - `split_table`:  해당 테이블을 W&B UI에서 별도의 섹션(이름: "Custom Chart Tables")으로 분리해 보여줄지 여부입니다. `True`로 설정하면 별도 섹션에 표시됩니다. 기본값은 `False`입니다. 



**리턴:**
 
 - `CustomChart`:  W&B에 로그할 수 있는 커스텀 차트 오브젝트입니다. 차트를 로그하려면 `wandb.log()`에 전달하세요. 



**예시:**
 

```python
import math
import random
import wandb

# 서로 다른 패턴의 여러 데이터 시리즈 생성
data = []
for i in range(100):
     # 시리즈 1: 무작위 노이즈가 더해진 사인 곡선 패턴
     data.append([i, math.sin(i / 10) + random.uniform(-0.1, 0.1), "series_1"])
     # 시리즈 2: 무작위 노이즈가 더해진 코사인 곡선 패턴
     data.append([i, math.cos(i / 10) + random.uniform(-0.1, 0.1), "series_2"])
     # 시리즈 3: 무작위 노이즈가 더해진 선형 증가 패턴
     data.append([i, i / 10 + random.uniform(-0.5, 0.5), "series_3"])

# 테이블에 사용할 컬럼 정의
table = wandb.Table(data=data, columns=["step", "value", "series"])

# wandb run을 시작하고 선 차트 로그하기
with wandb.init(project="line_chart_example") as run:
     line_chart = wandb.plot.line(
         table=table,
         x="step",
         y="value",
         stroke="series",  # "series" 컬럼별로 그룹화
         title="Multi-Series Line Plot",
     )
     run.log({"line-chart": line_chart})
```