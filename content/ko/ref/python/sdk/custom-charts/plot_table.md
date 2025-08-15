---
title: plot_table()
data_type_classification: function
menu:
  reference:
    identifier: ko-ref-python-sdk-custom-charts-plot_table
object_type: python_sdk_custom_charts
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/plot/custom_chart.py >}}




### <kbd>function</kbd> `plot_table`

```python
plot_table(
    vega_spec_name: 'str',
    data_table: 'wandb.Table',
    fields: 'dict[str, Any]',
    string_fields: 'dict[str, Any] | None' = None,
    split_table: 'bool' = False
) → CustomChart
```

Vega-Lite 명세와 `wandb.Table`을 사용하여 커스텀 차트를 생성합니다.

이 함수는 Vega-Lite 명세와 `wandb.Table` 오브젝트로 표현된 데이터 테이블을 기반으로 커스텀 차트를 만듭니다. 명세는 미리 정의되어 있고 W&B 백엔드에 저장되어 있어야 합니다. 이 함수는 커스텀 차트 오브젝트를 반환하며, 해당 오브젝트는 `wandb.Run.log()`를 사용해 W&B에 로그를 남길 수 있습니다.



**인수:**
 
 - `vega_spec_name`: 시각화 구조를 정의하는 Vega-Lite 명세의 이름 또는 식별자입니다.
 - `data_table`: 시각화할 데이터를 포함하고 있는  `wandb.Table` 오브젝트입니다.
 - `fields`: Vega-Lite 명세에서 사용하는 필드와 시각화할 데이터 테이블의 컬럼 간의 매핑입니다.
 - `string_fields`: 커스텀 시각화에 필요한 문자열 상수 값을 제공하기 위한 사전입니다.
 - `split_table`: 테이블을 W&B UI에서 별도의 섹션에 분리할지 여부입니다. `True`로 설정하면 "Custom Chart Tables"라는 섹션에 해당 테이블이 표시됩니다. 기본값은 `False`입니다.



**반환:**
 
 - `CustomChart`: W&B에 로그할 수 있는 커스텀 차트 오브젝트입니다. 차트를 로그하려면 오브젝트를 `wandb.Run.log()`의 인수로 전달하세요.



**예외:**
 
 - `wandb.Error`: `data_table`이 `wandb.Table` 오브젝트가 아닐 경우 발생합니다.



**예시:**
 ```python
# Vega-Lite 명세와 데이터 테이블을 이용하여 커스텀 차트를 생성합니다.
import wandb

data = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
table = wandb.Table(data=data, columns=["x", "y"])
fields = {"x": "x", "y": "y", "title": "MY TITLE"}

with wandb.init() as run:
    # 트레이닝 코드가 들어가는 부분입니다

    # `string_fields`로 커스텀 타이틀을 생성합니다.
    my_custom_chart = wandb.plot_table(
         vega_spec_name="wandb/line/v0",
         data_table=table,
         fields=fields,
         string_fields={"title": "Title"},
    )

    run.log({"custom_chart": my_custom_chart})
```