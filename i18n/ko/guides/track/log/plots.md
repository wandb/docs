---
description: Create and track plots from machine learning experiments.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 로그 플롯

<head>
  <title>W&B 실험에서 로그와 트랙 플롯.</title>
</head>

`wandb.plot`의 메서드를 사용하면 학습 동안 시간이 지남에 따라 변경되는 차트를 포함하여 `wandb.log`로 차트를 추적할 수 있습니다. 커스텀 차트 프레임워크에 대해 자세히 알아보려면 [이 가이드](../../app/features/custom-charts/walkthrough.md)를 확인하세요.

### 기본 차트

이 간단한 차트는 메트릭과 결과의 기본 시각화를 구성하기 쉽게 해줍니다.

<Tabs
  defaultValue="line"
  values={[
    {label: '선', value: 'line'},
    {label: '산점도', value: 'scatter'},
    {label: '막대', value: 'bar'},
    {label: '히스토그램', value: 'histogram'},
    {label: '다중선', value: 'multiline'},
  ]}>
  <TabItem value="line">

`wandb.plot.line()`

임의의 축 위에 연결되고 정렬된 점들의 목록인 커스텀 선형 플롯을 로그합니다.

```python
data = [[x, y] for (x, y) in zip(x_values, y_values)]
table = wandb.Table(data=data, columns=["x", "y"])
wandb.log(
    {
        "my_custom_plot_id": wandb.plot.line(
            table, "x", "y", title="Custom Y vs X 선형 플롯"
        )
    }
)
```

두 차원에서 곡선을 로그하는 데 이를 사용할 수 있습니다. 두 값 목록을 서로 대응시켜 플로팅하는 경우, 목록의 값 수가 정확히 일치해야 합니다(즉, 각 점은 x와 y를 가져야 함).

![](/images/track/line_plot.png)

[앱에서 보기 →](https://wandb.ai/wandb/plots/reports/Custom-Line-Plots--VmlldzoyNjk5NTA)

[코드 실행 →](https://tiny.cc/custom-charts)
  </TabItem>
  <TabItem value="scatter">

`wandb.plot.scatter()`

임의의 축 x와 y에 대한 점(x, y) 목록인 커스텀 산점도를 로그합니다.

```python
data = [[x, y] for (x, y) in zip(class_x_scores, class_y_scores)]
table = wandb.Table(data=data, columns=["class_x", "class_y"])
wandb.log({"my_custom_id": wandb.plot.scatter(table, "class_x", "class_y")})
```

두 차원에서 산점도 점을 로그하는 데 이를 사용할 수 있습니다. 두 값 목록을 서로 대응시켜 플로팅하는 경우, 목록의 값 수가 정확히 일치해야 합니다(즉, 각 점은 x와 y를 가져야 함).

![](/images/track/demo_scatter_plot.png)

[앱에서 보기 →](https://wandb.ai/wandb/plots/reports/Custom-Scatter-Plots--VmlldzoyNjk5NDQ)

[코드 실행 →](https://tiny.cc/custom-charts)
  </TabItem>
  <TabItem value="bar">

`wandb.plot.bar()`

막대로 표현된 레이블이 지정된 값 목록인 커스텀 막대 차트를 몇 줄로 로그합니다:

```python
data = [[label, val] for (label, val) in zip(labels, values)]
table = wandb.Table(data=data, columns=["label", "value"])
wandb.log(
    {
        "my_bar_chart_id": wandb.plot.bar(
            table, "label", "value", title="Custom 막대 차트"
        )
    }
)
```

임의의 막대 차트를 로그하는 데 이를 사용할 수 있습니다. 목록의 레이블과 값 수가 정확히 일치해야 합니다(즉, 각 데이터 포인트는 둘 다 있어야 함).

![](/images/track/basic_charts_bar.png)

[앱에서 보기 →](https://wandb.ai/wandb/plots/reports/Custom-Bar-Charts--VmlldzoyNzExNzk)

[코드 실행 →](https://tiny.cc/custom-charts)
  </TabItem>
  <TabItem value="histogram">

`wandb.plot.histogram()`

발생 빈도/횟수에 따라 값 목록을 구간으로 정렬하는 커스텀 히스토그램을 몇 줄로 로그합니다. 예를 들어 예측 신뢰도 점수(`scores`) 목록이 있고 그 분포를 시각화하고 싶다고 가정해 보겠습니다:

```python
data = [[s] for s in scores]
table = wandb.Table(data=data, columns=["scores"])
wandb.log({"my_histogram": wandb.plot.histogram(table, "scores", title="히스토그램")})
```

임의의 히스토그램을 로그하는 데 이를 사용할 수 있습니다. `data`는 행과 열의 2D 배열을 지원하기 위해 의도된 목록의 목록입니다.

![](/images/track/demo_custom_chart_histogram.png)

[앱에서 보기 →](https://wandb.ai/wandb/plots/reports/Custom-Histograms--VmlldzoyNzE0NzM)

[코드 실행 →](https://tiny.cc/custom-charts)
  </TabItem>
  <TabItem value="multiline">

`wandb.plot.line_series()`

하나의 공유된 x-y 축 세트에서 x-y 좌표 쌍의 여러 목록 또는 여러 줄을 플로팅합니다:

```python
wandb.log(
    {
        "my_custom_id": wandb.plot.line_series(
            xs=[0, 1, 2, 3, 4],
            ys=[[10, 20, 30, 40, 50], [0.5, 11, 72, 3, 41]],
            keys=["metric Y", "metric Z"],
            title="두 개의 랜덤 메트릭",
            xname="x 단위",
        )
    }
)
```

x와 y 점의 수가 정확히 일치해야 합니다. 여러 y 값 목록에 맞는 하나의 x 값 목록을 제공하거나 각 y 값 목록에 대해 별도의 x 값 목록을 제공할 수 있습니다.

![](/images/track/basic_charts_histogram.png)

[앱에서 보기 →](https://wandb.ai/wandb/plots/reports/Custom-Multi-Line-Plots--VmlldzozOTMwMjU)
  </TabItem>
</Tabs>

### 모델 평가 차트

이 프리셋 차트는 스크립트에서 직접 차트를 로그하고 UI에서 원하는 정보를 쉽고 빠르게 볼 수 있도록 내장된 `wandb.plot` 메서드를 사용합니다.

<Tabs
  defaultValue="precision_recall"
  values={[
    {label: 'Precision-Recall 곡선', value: 'precision_recall'},
    {label: 'ROC 곡선', value: 'roc'},
    {label: '혼동 행렬', value: 'confusion_matrix'},
  ]}>
  <TabItem value="precision_recall">

`wandb.plot.pr_curve()`

한 줄로 [Precision-Recall 곡선](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision\_recall\_curve.html#sklearn.metrics.precision\_recall\_curve)을 생성합니다:

```python
wandb.log({"pr": wandb.plot.pr_curve(실제값, 예측값)})
```

코드가 다음에 액세스할 수 있을 때마다 이를 로그할 수 있습니다:

* 한 세트의 예제에 대한 모델의 예측 점수(`예측값`)
* 그 예제들에 대한 해당 실제값 레이블(`실제값`)
* (선택사항) 레이블/클래스 이름 목록(`labels=["cat", "dog", "bird"...]` 경우 레이블 인덱스 0은 고양이, 1=강아지, 2=새 등)
* (선택사항) 플롯에서 시각화할 레이블의 서브세트(여전히 목록 형식)

![](/images/track/model_eval_charts_precision_recall.png)

[앱에서 보기 →](https://wandb.ai/wandb/plots/reports/Plot-Precision-Recall-Curves--VmlldzoyNjk1ODY)

[코드 실행 →](https://colab.research.google.com/drive/1mS8ogA3LcZWOXchfJoMrboW3opY1A8BY?usp=sharing)
  </TabItem>
  <TabItem value="roc">

`wandb.plot.roc_curve()`

한 줄로 [ROC 곡선](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc\_curve.html#sklearn.metrics.roc\_curve)을 생성합니다:

```python
wandb.log({"roc": wandb.plot.roc_curve(실제값, 예측값)})
```

코드가 다음에 액세스할 수 있을 때마다 이를 로그할 수 있습니다:

* 한 세트의 예제에 대한 모델의 예측 점수(`예측값`)
* 그 예제들에 대한 해당 실제값 레이블(`실제값`)
* (선택사항) 레이블/클래스 이름 목록(`labels=["cat", "dog", "bird"...]` 경우 레이블 인덱스 0은 고양이, 1=강아지, 2=새 등)
* (선택사항) 플롯에서 시각화할 이 레이블의 서브세트(여전히 목록 형식)

![](/images/track/demo_custom_chart_roc_curve.png)

[앱에서 보기 →](https://wandb.ai/wandb/plots/reports/Plot-ROC-Curves--VmlldzoyNjk3MDE)

[코드 실행 →](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Plot\_ROC\_Curves\_with\_W%26B.ipynb)
  </TabItem>
  <TabItem value="confusion_matrix">

`wandb.plot.confusion_matrix()`

한 줄로 다중 클래스 [혼동 행렬](https://scikit-learn.org/stable/auto\_examples/model\_selection/plot\_confusion\_matrix.html)을 생성합니다:

```python
cm = wandb.plot.confusion_matrix(
    y_true=실제값, preds=예측값, class_names=class_names
)

wandb.log({"conf_mat": cm})
```

코드가 다음에 액세스할 수 있을 때마다 이를 로그할 수 있습니다:

* 한 세트의 예제에 대한 모델의 예측 레이블(`preds`) 또는 정규화된 확률 점수(`probs`). 확률은 (예제 수, 클래스 수)의 형태를 가져야 합니다. 확률 또는 예측 중 하나를 제공할 수 있지만 둘 다는 안 됩니다.
* 그 예제들에 대한 해당 실제값 레이블(`y_true`)
* 문자열로 된 레이블/클래스 이름의 전체 목록(`class_names`, 예: `class_names=["cat", "dog", "bird"]` 경우 인덱스 0은 고양이, 1=강아지, 2=새 등)

![](/images/experiments/confusion_matrix.png)

[앱에서 보기 →](https://wandb.ai/wandb/plots/reports/Confusion-Matrix--VmlldzozMDg1NTM)​

[코드 실행 →](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Log\_a\_Confusion\_Matrix\_with\_W%26B.ipynb)
  </TabItem>
</Tabs>

### 인터랙티브 커스텀 차트

전체 사용자 정의를 위해 내장된 [커스텀 차트 프리셋](../../app/features/custom-charts/walkthrough.md)을 조정하거나 새 프리셋을 생성한 다음 차트를 저장하세요. 스크립트에서 직접 해당 커스텀 프리셋에 데이터를 로그하려면 차트 ID를 사용하세요.

```python
# 플롯할 열이 있는 테이블 생성
table = wandb.Table(data=data, columns=["step", "height"])

# 테이블의 열을 차트의 필드에 매핑
fields = {"x": "step", "value": "height"}

# 테이블을 사용하여 새로운 커스텀 차트 프리셋 채우기
# 자신의 저장된 차트 프리셋을 사용하려면 vega_spec_name 변경
# 제목을 편집하려면 string_fields 변경
my_custom_chart = wandb.plot_table(
    vega_spec_name="carey/new_chart",
    data_table=table,
    fields=fields,
    string_fields={"title": "높이 히스토그램"},
)
```

[코드 실행 →](https://tiny.cc/custom-charts)

### Matplotlib 및 Plotly 플롯

`wandb.plot`과 함께 W&B [커스텀 차트](../../app/features/custom-charts/walkthrough.md)를 사용하는 대신, [matplotlib](https://matplotlib.org/) 및 [Plotly](https://plotly.com/)로 생성된 차트를 로그할 수 있습니다.

```python
import matplotlib.pyplot as plt

plt.plot([1, 2, 3, 4])
plt.ylabel("몇몇 흥미로운 숫자들")
wandb.log({"chart": plt})
```

`matplotlib` 플롯 또는 피규어 객체를 `wandb.log()`에 전달하기만 하면 됩니다. 기본적으로 우리는 플롯을 [Plotly](https://plot.ly/) 플롯으로 변환합니다. 플롯을 이미지로 로그하려면 플롯을 `wandb.Image`로 전달할 수 있습니다. Plotly 차트도 직접 받습니다.

:::info
"빈 플롯을 로그하려고 시도했습니다"라는 오류가 발생하는 경우 `fig = plt.figure()`로 피규어를 플롯과 별도로 저장한 다음 `wandb.log` 호출에서 `fig`를 로그할 수 있습니다.
:::

### W&B 테이블에 커스텀 HTML 로그

W&B는 Plotly 및 Bokeh에서 HTML로 인터랙티브 차트를 로그하고 테이블에 추가하는 것을 지원합니다.

#### 테이블에 Plotly 그림을 HTML로 로그

Plotly 인터랙티브 차트를 HTML로 변환하여 wandb 테이블에 로그할 수 있습니다.

```python
import wandb
import plotly.express as px

# 새 실행 초기화
run = wandb.init(project="log-plotly-fig-tables", name="plotly_html")

# 테이블 생성
table = wandb.Table(columns=["plotly_figure"])

# Plotly 그림 경로 생성
path_to_plotly_html = "./plotly_figure.html"

# 예시 Plotly 그림
fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])

# Plotly 그림을 HTML로 쓰기
# auto_play를 False로 설정하면 Plot