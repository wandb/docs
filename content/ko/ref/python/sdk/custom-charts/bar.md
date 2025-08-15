---
title: bar()
data_type_classification: function
menu:
  reference:
    identifier: ko-ref-python-sdk-custom-charts-bar
object_type: python_sdk_custom_charts
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/plot/bar.py >}}




### <kbd>function</kbd> `bar`

```python
bar(
    table: 'wandb.Table',
    label: 'str',
    value: 'str',
    title: 'str' = '',
    split_table: 'bool' = False
) → CustomChart
```

wandb.Table 에 담긴 데이터를 사용해 막대그래프(Bar Chart)를 생성합니다. 



**인자:**
 
 - `table`:  막대그래프에 사용할 데이터를 담은 테이블입니다. 
 - `label`:  각 막대의 라벨에 사용할 컬럼 이름입니다. 
 - `value`:  각 막대의 값에 사용할 컬럼 이름입니다. 
 - `title`:  막대그래프의 제목입니다. 
 - `split_table`:  이 값을 `True` 로 지정하면, 테이블이 W&B UI에서 별도의 "Custom Chart Tables" 섹션에 표시됩니다. 기본값은 `False` 입니다. 



**반환값:**
 
 - `CustomChart`:  W&B로 로그할 수 있는 커스텀 차트 오브젝트입니다. 차트를 로그하려면 `wandb.log()` 에 전달하세요. 



**예시:**
 

```python
import random
import wandb

# 테이블에 사용할 랜덤 데이터 생성
data = [
    ["car", random.uniform(0, 1)],
    ["bus", random.uniform(0, 1)],
    ["road", random.uniform(0, 1)],
    ["person", random.uniform(0, 1)],
]

# 데이터로 table 생성
table = wandb.Table(data=data, columns=["class", "accuracy"])

# W&B run을 시작하고, bar plot을 로그합니다.
with wandb.init(project="bar_chart") as run:
    # 테이블로부터 bar plot 생성
    bar_plot = wandb.plot.bar(
         table=table,
         label="class",
         value="accuracy",
         title="Object Classification Accuracy",
    )

    # bar plot을 W&B에 로그
    run.log({"bar_plot": bar_plot})
```