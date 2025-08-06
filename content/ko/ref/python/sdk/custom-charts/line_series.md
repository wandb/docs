---
title: 'line_series()

  '
data_type_classification: function
menu:
  reference:
    identifier: ko-ref-python-sdk-custom-charts-line_series
object_type: python_sdk_custom_charts
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/plot/line_series.py >}}




### <kbd>function</kbd> `line_series`

```python
line_series(
    xs: 'Iterable[Iterable[Any]] | Iterable[Any]',
    ys: 'Iterable[Iterable[Any]]',
    keys: 'Iterable[str] | None' = None,
    title: 'str' = '',
    xname: 'str' = 'x',
    split_table: 'bool' = False
) → CustomChart
```

라인 시리즈 차트를 생성합니다.



**인수:**

 - `xs`:  x 값의 시퀀스입니다. 하나의 배열이 제공되면 모든 y 값이 해당 x 배열에 대해 플롯됩니다. 배열의 배열이 제공되면, 각각의 y 값이 해당하는 x 배열에 대해 플롯됩니다.
 - `ys`:  y 값의 시퀀스이며, 각 반복 가능한 객체가 별도의 라인 시리즈를 나타냅니다.
 - `keys`:  각 라인 시리즈에 라벨을 붙이기 위한 키의 시퀀스입니다. 제공되지 않으면, "line_1", "line_2" 등으로 자동 생성됩니다.
 - `title`:  차트의 제목입니다.
 - `xname`:  x축의 라벨 이름입니다.
 - `split_table`:  테이블을 W&B UI에서 별도의 섹션에 분리해서 보여줄지 여부입니다. `True`로 설정하면, 테이블이 "Custom Chart Tables"라는 섹션에 표시됩니다. 기본값은 `False`입니다.



**반환값:**

 - `CustomChart`:  W&B에 로그할 수 있는 커스텀 차트 오브젝트입니다. 차트를 로그하려면 `wandb.log()`에 전달하세요.



**예시:**
하나의 x 배열로 모든 y 시리즈를 동일한 x 값에 대해 플롯하는 방법:

```python
import wandb

# W&B run 초기화
with wandb.init(project="line_series_example") as run:
    # 모든 y 시리즈에 공통적으로 사용할 x 값 배열
    xs = list(range(10))

    # 여러 y 시리즈
    ys = [
         [i for i in range(10)],  # y = x
         [i**2 for i in range(10)],  # y = x^2
         [i**3 for i in range(10)],  # y = x^3
    ]

    # 라인 시리즈 차트 생성 및 로그
    line_series_chart = wandb.plot.line_series(
         xs,
         ys,
         title="title",
         xname="step",
    )
    run.log({"line-series-single-x": line_series_chart})
```

이 예시에서는 하나의 `xs` 시리즈(공통 x 값)를 모든 `ys` 시리즈에 사용합니다. 그래서 각 y 시리즈가 동일한 x 값(0-9)에 대해 그려집니다.

각각의 y 시리즈가 자신만의 x 배열에 대해 그려지는 예시:

```python
import wandb

# W&B run 초기화
with wandb.init(project="line_series_example") as run:
    # 각 y 시리즈에 대한 서로 다른 x 값 배열
    xs = [
         [i for i in range(10)],  # 첫 번째 시리즈용 x
         [2 * i for i in range(10)],  # 두 번째 시리즈용 x (늘림)
         [3 * i for i in range(10)],  # 세 번째 시리즈용 x (더 늘림)
    ]

    # 해당하는 y 시리즈
    ys = [
         [i for i in range(10)],  # y = x
         [i**2 for i in range(10)],  # y = x^2
         [i**3 for i in range(10)],  # y = x^3
    ]

    # 라인 시리즈 차트 생성 및 로그
    line_series_chart = wandb.plot.line_series(
         xs, ys, title="Multiple X Arrays Example", xname="Step"
    )
    run.log({"line-series-multiple-x": line_series_chart})
```

이 예시에서는 각각의 y 시리즈가 각기 다른 x 시리즈에 대해 그려집니다. 데이터 시리즈별로 x 값이 다를 경우 더 유연하게 사용할 수 있습니다.

`keys` 인수를 이용해 라인별 라벨을 커스터마이즈 하는 방법:

```python
import wandb

# W&B run 초기화
with wandb.init(project="line_series_example") as run:
    xs = list(range(10))  # 하나의 x 배열
    ys = [
         [i for i in range(10)],  # y = x
         [i**2 for i in range(10)],  # y = x^2
         [i**3 for i in range(10)],  # y = x^3
    ]

    # 각 라인에 대한 커스텀 라벨
    keys = ["Linear", "Quadratic", "Cubic"]

    # 라인 시리즈 차트 생성 및 로그
    line_series_chart = wandb.plot.line_series(
         xs,
         ys,
         keys=keys,  # 커스텀 키 (라인 라벨)
         title="Custom Line Labels Example",
         xname="Step",
    )
    run.log({"line-series-custom-keys": line_series_chart})
```

이 예시에서는 `keys` 인수를 사용하여 라인별로 커스텀 라벨을 지정하는 방법을 보여줍니다. 범례에는 "Linear", "Quadratic", "Cubic"이 표시됩니다.